#include <atomic>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/core/mppi_common.cuh>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#define VANILLA_MPPI_TEMPLATE                                                                                          \
  template <class DYN_T, class COST_T, class FB_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T,              \
            class PARAMS_T>

#define VanillaMPPI VanillaMPPIController<DYN_T, COST_T, FB_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

VANILLA_MPPI_TEMPLATE
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt,
                                   int max_iter, float lambda, float alpha, int num_timesteps,
                                   const Eigen::Ref<const control_trajectory>& init_control_traj, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, init_control_traj,
                 stream)
{
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Copy the noise std_dev to the device
  // this->copyControlStdDevToDevice();

  chooseAppropriateKernel();
}

VANILLA_MPPI_TEMPLATE
VanillaMPPI::VanillaMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler,
                                   PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
{
  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // // Copy the noise std_dev to the device
  // this->copyControlStdDevToDevice();
  chooseAppropriateKernel();
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::chooseAppropriateKernel()
{
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
  unsigned single_kernel_byte_size = mppi::kernels::calcRolloutCombinedKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_dyn_kernel_byte_size = mppi::kernels::calcRolloutDynamicsKernelSharedMemSize(
      this->model_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_cost_kernel_byte_size =
      mppi::kernels::calcRolloutCostKernelSharedMemSize(this->cost_, this->sampler_, this->params_.cost_rollout_dim_);
  unsigned vis_single_kernel_byte_size = mppi::kernels::calcVisualizeKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->getNumTimesteps(), this->params_.visualize_dim_);

  bool too_much_mem_single_kernel = single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_vis_kernel = vis_single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_split_kernel = split_dyn_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_split_kernel = too_much_mem_split_kernel || split_cost_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_single_kernel = too_much_mem_single_kernel || too_much_mem_vis_kernel;

  if (too_much_mem_split_kernel && too_much_mem_single_kernel)
  {
    std::string error_msg =
        "There is not enough shared memory on the GPU for either rollout kernel option. The combined rollout kernel "
        "takes " +
        std::to_string(single_kernel_byte_size) + " bytes, the cost rollout kernel takes " +
        std::to_string(split_cost_kernel_byte_size) + " bytes, the dynamics rollout kernel takes " +
        std::to_string(split_dyn_kernel_byte_size) + " bytes, the combined visualization kernel takes " +
        std::to_string(vis_single_kernel_byte_size) + " bytes, and the max is " +
        std::to_string(deviceProp.sharedMemPerBlock) +
        " bytes. Considering lowering the corresponding thread block sizes.";
    throw std::runtime_error(error_msg);
  }
  else if (too_much_mem_single_kernel)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    return;
  }
  else if (too_much_mem_split_kernel)
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    return;
  }

  // Send the nominal control to the device
  this->copyNominalControlToDevice(false);
  state_array zero_state = this->model_->getZeroState();
  // Send zero state to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, zero_state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));
  // Generate noise data
  this->sampler_->generateSamples(1, 0, this->gen_, true);

  float single_kernel_time_ms = std::numeric_limits<float>::infinity();
  float split_kernel_time_ms = std::numeric_limits<float>::infinity();

  // Evaluate each kernel that is applicable
  auto start_single_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_single_kernel; i++)
  {
    mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
        this->params_.dynamics_rollout_dim_, this->stream_, true);
  }
  auto end_single_kernel_time = std::chrono::steady_clock::now();
  auto start_split_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_split_kernel; i++)
  {
    mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
        this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_, this->stream_, true);
  }
  auto end_split_kernel_time = std::chrono::steady_clock::now();

  // calc times
  if (!too_much_mem_single_kernel)
  {
    single_kernel_time_ms = mppi::math::timeDiffms(end_single_kernel_time, start_single_kernel_time);
  }
  if (!too_much_mem_split_kernel)
  {
    split_kernel_time_ms = mppi::math::timeDiffms(end_split_kernel_time, start_split_kernel_time);
  }
  std::string kernel_choice = "";
  if (split_kernel_time_ms < single_kernel_time_ms)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    kernel_choice = "split ";
  }
  else
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    kernel_choice = "single";
  }
  this->logger_->info("Choosing %s kernel based on split taking %f ms and single taking %f ms after %d iterations\n",
                     kernel_choice.c_str(), split_kernel_time_ms, single_kernel_time_ms,
                     this->getNumKernelEvaluations());
}

VANILLA_MPPI_TEMPLATE
VanillaMPPI::~VanillaMPPIController()
{
  // destructor 
  // all implemented in standard controller
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost();

  // Send the initial condition to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_)); // (dst, src, ...)

  float baseline_prev = 1e8;

  for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++) // num_iters_ default is 1
  {
    // Send the nominal control to the device
    this->copyNominalControlToDevice(false); // send: control_

    // Generate noise data
    this->sampler_->generateSamples(optimization_stride, opt_iter, this->gen_, false);

    // Launch the rollout kernel
    if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS)
    {
      mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
          this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
          this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
          this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_, this->stream_, false);
    }
    else if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
    {
      mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
          this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
          this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
          this->params_.dynamics_rollout_dim_, this->stream_, false);
    }

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_)); // this->trajectory_costs_: cost for every sampled trajectory (Mx1)

    this->setBaseline(mppi::kernels::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS)); // the minimum cost

    if (this->getBaselineCost() > baseline_prev + 1)
    {
      this->logger_->debug("Previous Baseline: %f\n         Baseline: %f\n", baseline_prev, this->getBaselineCost());
    }

    baseline_prev = this->getBaselineCost();

    // Launch the norm exponential kernel
    mppi::kernels::launchNormExpKernel(NUM_ROLLOUTS, this->getNormExpThreads(), this->trajectory_costs_d_,
                                       1.0 / this->getLambda(), this->getBaselineCost(), this->stream_, false);
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->setNormalizer(mppi::kernels::computeNormalizer(this->trajectory_costs_.data(), NUM_ROLLOUTS));

    mppi::kernels::computeFreeEnergy(this->free_energy_statistics_.real_sys.freeEnergyMean,
                                     this->free_energy_statistics_.real_sys.freeEnergyVariance,
                                     this->free_energy_statistics_.real_sys.freeEnergyModifiedVariance,
                                     this->trajectory_costs_.data(), NUM_ROLLOUTS, this->getBaselineCost(),
                                     this->getLambda());

    this->sampler_->updateDistributionParamsFromDevice(this->trajectory_costs_d_, this->getNormalizerCost(), 0, false);

    // Transfer the new control to the host
    this->sampler_->setHostOptimalControlSequence(this->control_.data(), 0, true);
  }

  this->free_energy_statistics_.real_sys.normalizerPercent = this->getNormalizerCost() / NUM_ROLLOUTS;
  this->free_energy_statistics_.real_sys.increase =
      this->getBaselineCost() - this->free_energy_statistics_.real_sys.previousBaseline;
  smoothControlTrajectory();
  computeStateTrajectory(state);
  state_array zero_state = this->model_->getZeroState();
  for (int i = 0; i < this->getNumTimesteps(); i++)
  {
    this->model_->enforceConstraints(zero_state, this->control_.col(i));
  }

  // Copy back sampled trajectories
  this->copySampledControlFromDevice(false);
  if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
  {  // copy initial state to vis initial state for use with visualizeKernel
    HANDLE_ERROR(cudaMemcpyAsync(this->vis_initial_state_d_, this->initial_state_d_, sizeof(float) * DYN_T::STATE_DIM,
                                 cudaMemcpyDeviceToDevice, this->vis_stream_));
  }
  this->copyTopControlFromDevice(true);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::allocateCUDAMemory()
{// allocate device memory for the initial state, control sequence, output trajectory, and sampled costs
  PARENT_CLASS::allocateCUDAMemoryHelper();
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0)
{
  this->computeOutputTrajectoryHelper(this->output_, this->state_, x0, this->control_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::smoothControlTrajectory()
{
  this->smoothControlTrajectoryHelper(this->control_, this->control_history_);
}

VANILLA_MPPI_TEMPLATE
void VanillaMPPI::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->getTotalSampledTrajectories(); // 2048
  // ROS_INFO("num_sampled_trajectories: %d\n", num_sampled_trajectories);
  // ROS_INFO("sampled_outputs_d_ pointer: %p", (void*)this->sampled_outputs_d_);
  // --- Debug: Copy and print initial state from device ---
  // HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  // const int state_dim = DYN_T::STATE_DIM;
  // std::vector<float> initial_state_host(state_dim, 0.0f);
  // // Copy the initial state from device to host
  // HANDLE_ERROR(cudaMemcpy(initial_state_host.data(), this->initial_state_d_, 
  //                         sizeof(float) * state_dim,
  //                         cudaMemcpyDeviceToHost));
  // std::stringstream ss;
  // ss << "Initial state from device: ";
  // for (int i = 0; i < state_dim; i++) {
  //     ss << initial_state_host[i] << " ";
  // }
  // ROS_INFO("%s", ss.str().c_str());

  // control already copied in compute control, so run kernel
  if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS) // choose the split kernel
  {
    mppi::kernels::launchVisualizeCostKernel<COST_T, SAMPLING_T>(
        this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->sampled_outputs_d_, this->sampled_crash_status_d_,
        this->sampled_costs_d_, this->params_.cost_rollout_dim_, this->stream_, false);
  }
  else if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
  {
    mppi::kernels::launchVisualizeKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->vis_initial_state_d_, this->sampled_outputs_d_,
        this->sampled_costs_d_, this->sampled_crash_status_d_, this->params_.visualize_dim_, this->stream_, false);
  }

  // // Optionally, synchronize and read a few state values from sampled_outputs_d_ directly if needed.
  // HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  // float first_state;
  // HANDLE_ERROR(cudaMemcpy(&first_state, this->sampled_outputs_d_, sizeof(float), cudaMemcpyDeviceToHost));
  // ROS_INFO("First element of sampled_outputs_d_ = %f", first_state);

  // note!!! based on copySampledControlFromDevice()
  // sampled_outputs_d_ saves the nominal trajecory starting with initial state in index 0
  // sampled_outputs_d_ saves other sampled trajectories starting from the second state in index 1 to the end
  // the difference is handled by adding: int t_intial = thread_idx + time_iter * blockDim.x; int t = (global_idx > 0) ? t_intial + 1 : t_intial;
  // if perceptage is above 0.98, the order of sampled trajectories is the same as generated in the controller kernel
  // the size if NUM_ROLLOUTS * NUM_TIMESTEPS * OUTPUT_DIM

  // then the cost and crash status are queried based on the state above
  // the returned sampled_outputs_ is the state from 0 to 98 for 0 trajectory, and 1 to 99 for other sampled trajectories
  // the returned sampled_costs_ is the cost from 1 to 99 and a terminal cost at the end for all sampled trajectories

  for (int i = 0; i < num_sampled_trajectories; i++)
  {
    // set initial state to the first location
    // shifted by one since we do not save the initial state
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_trajectories_[i].data(),
                                 this->sampled_outputs_d_ + i * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
                                 (this->getNumTimesteps() - 1) * DYN_T::OUTPUT_DIM * sizeof(float),
                                 cudaMemcpyDeviceToHost, this->vis_stream_));
    // HANDLE_ERROR(
    //     cudaMemcpyAsync(this->sampled_costs_[i].data(), this->sampled_costs_d_ + (i * (this->getNumTimesteps() + 1)),
    //                     (this->getNumTimesteps() + 1) * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(
      cudaMemcpyAsync(this->sampled_costs_[i].data(), this->sampled_costs_d_ + (i * (this->getNumTimesteps())),
                      (this->getNumTimesteps()) * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_)); // modified with visualizeCostKernel to avoid misalignment
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_crash_status_[i].data(),
                                 this->sampled_crash_status_d_ + (i * this->getNumTimesteps()),
                                 this->getNumTimesteps() * sizeof(int), cudaMemcpyDeviceToHost, this->vis_stream_));
    // if (i < 2) { // only print for first two trajectories to avoid flooding
    //   cudaStreamSynchronize(this->vis_stream_);  // wait for memcpy to finish for this trajectory
    //   std::stringstream ss;
    //   ss << "Trajectory " << i << " cost values: ";
    //   for (int t = 0; t < this->getNumTimesteps(); t++) {
    //     ss << this->sampled_costs_[i](t, 0) << " ";
    //   }
    //   ROS_INFO("%s", ss.str().c_str());
    // }
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
}

#undef VANILLA_MPPI_TEMPLATE
#undef VanillaMPPI
