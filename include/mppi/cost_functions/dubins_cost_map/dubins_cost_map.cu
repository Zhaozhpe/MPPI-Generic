#include "dubins_cost_map.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// ----------------------
// Constructor
// ----------------------
template <class CLASS_T, class DYN_T, class PARAMS_T>
DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::DubinsCostMapImpl(cudaStream_t stream)
{
  this->bindToStream(stream);
}

// ----------------------
// Host Functions
// ----------------------
template <class CLASS_T, class DYN_T, class PARAMS_T>
float DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(const Eigen::Ref<const output_array> s,
                                                                     int timestep, int* crash_status)
{
  float cost = 0.0f;
  output_array desired_state = this->params_.getDesiredState(timestep);
  // Quadratic state error.
  Eigen::Matrix<float, DYN_T::OUTPUT_DIM, DYN_T::OUTPUT_DIM> coeffs;
  coeffs.setZero();
  for (int i = 0; i < DYN_T::OUTPUT_DIM; i++) {
    coeffs(i, i) = this->params_.s_coeffs[i];
  }
  output_array error = s - desired_state;
  cost = error.transpose() * coeffs * error;

  // Add map cost: assume vehicle position is already used to build the grid.
  float map_cost = queryTextureTransformed(s[0], s[1]);
  cost += this->params_.map_cost_weight * fabsf(map_cost);
  // float sampled_v = u[0];
  // float4 texf = queryTextureTransformed(s[0], s[1]);
  // float A_val     = texf.x; // “amplitude”
  // float mu_val    = texf.y;
  // float sigma_val = texf.z;

  // sigma_val = fmaxf(sigma_val, 1e-3f);
  // float diff = sampled_v - mu_val;
  // float log_term = 0.5f * logf(2.0f * M_PI * sigma_val * sigma_val);
  // float sq_term  = (diff * diff) / (2.0f * sigma_val * sigma_val);
  // float nll = sq_term + log_term;

  // // 3e) Scale by A and by map_cost_weight:
  // map_cost = A_val * nll * this->params_.map_cost_weight;
  // cost += map_cost;

  if (fabsf(map_cost) > 500.0f) {
    float crash_penalty = 10000000.0f; // A very large penalty.
    cost += crash_penalty;
    if (crash_status != nullptr)
      crash_status[0] = 1;
  }

  return cost;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
float DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(const Eigen::Ref<const output_array> s)
{
  return 0.0f;
}

// ----------------------
// Device Functions
// ----------------------

template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::computeRunningCost(float* y, float* u, int timestep, float* theta_c, int* crash)
{
  if (threadIdx.y == 0)
  {
    // CLASS_T* derived = static_cast<CLASS_T*>(this);
    return this->computeStateCost(y, u, timestep, theta_c, crash);
  }
  else
  {
    return 0.0f;
  }
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(float* s, float* u, int timestep, float* theta_c, int* crash_status)
{
  // query timestep: 0 to 99
  float* desired_state = this->params_.getGoalStatePointer();

  const int T = this->params_.timesteps_total;
  float t_norm = float(timestep) / float(T - 1);        // in [0,1]
  float w_t     = powf(t_norm, 2.0f);

  float distance_error = 0.0f;
  // float curvature_error = 0.0f;
  float omega_cost = 0.0f;
  if (timestep >= (this->params_.timesteps_total - 100)){
    float dx = s[0] - desired_state[0];
    float dy = s[1] - desired_state[1];
    // float d_k = s[4] - 0.0f; // curvature rate, s[4] is the curvature rate in the state vector.
    float omega = s[3] * s[4];
    float base_sq = dx*dx * this->params_.s_coeffs[0] + dy*dy * this->params_.s_coeffs[1];
    distance_error = w_t * sqrtf(base_sq);
    omega_cost = omega*omega * this->params_.s_coeffs[4];
    // curvature_error = d_k * d_k * this->params_.s_coeffs[4];
  } //w_t * 

  // Terminal cost: only at the last timestep.
  float terminal = 0.0f;
  if (timestep == T-1) {
    float dx = s[0] - desired_state[0];
    float dy = s[1] - desired_state[1];
    terminal = this->params_.terminal_cost_weight * (dx*dx + dy*dy);
  }

  // Yaw error only considered at final several timestep
  float yaw_error = 0.0f;
  if (timestep == (this->params_.timesteps_total - 10)) {
      float dyaw = s[2] - desired_state[2];
      // Normalize yaw error between [-pi, pi] if necessary
      dyaw = atan2f(sinf(dyaw), cosf(dyaw));
      yaw_error = dyaw * dyaw * this->params_.s_coeffs[2]; 
  }

  float state_cost = distance_error + yaw_error + omega_cost;

  // Control cost: quadratic in acceleration and curvature rate.
  float accel = u[0];
  float curvature_rate = u[1];
  float ctrl_cost = accel * accel * this->params_.control_cost_coeff[0]
                  + curvature_rate * curvature_rate * this->params_.control_cost_coeff[1];

  // Map cost: query the texture for the map cost at the current position.
  float vel = s[3];
  // float vel = 3.0f; // Assume a constant velocity for the map cost query.
  float4 texf = queryTextureTransformed(s[0], s[1]);
  float A_val     = texf.x; // “amplitude”
  float mu_val    = texf.y;
  float sigma_val = texf.z;
  sigma_val = fmaxf(sigma_val, 1e-3f);

  float map_cost = 0.0f;

  if (this->params_.adaptive_cost) {
    float diff = vel - mu_val;
    float exponent = - (diff * diff) / (2.0f * sigma_val * sigma_val);
    map_cost = A_val * expf(exponent);
  }
  else {
    map_cost = A_val;
  }

  float obs_cost  = 0.0f;

  if (map_cost <= 0.3f) {
    obs_cost = map_cost * this->params_.map_cost_weight[0];

  // medium‐risk band: (0.3, 0.5]
  } else if (map_cost <= 0.5f) {
    obs_cost = map_cost * this->params_.map_cost_weight[1];

  // high‐risk band: (0.5, 1.0]
  } else {
    obs_cost = map_cost * this->params_.map_cost_weight[2];
    if (crash_status) crash_status[0] = 1;
  }

  // float cost = state_cost + obs_cost + crash_penalty + ctrl_cost; 
  float cost = state_cost + ctrl_cost + obs_cost + terminal;

  // if (blockIdx.x == 0) {
  //   // Print the this->params_.... and values for debugging.
  //   printf(
  //     "\n%d:v=%.2f,k=%.2f,m=%.2f,x=%.3f,y=%.3f, A=%.3f mu=%.3f sigma=%.3f\n"
  //     "[s=(d=%.2f+w=%.2f)=%.2f+ctr=%.2f+te=%.2f+ob=%.2f]=%.2f\n",
  //     timestep, s[3], s[4], map_cost, s[0], s[1], A_val, mu_val, sigma_val,
  //     distance_error, omega_cost, state_cost, ctrl_cost, terminal, obs_cost, cost
  //   );
  // }

  // if (blockIdx.x == 0) {
  //   // Print a single line:
  //   printf(
  //     "DBG [block (%d,%d), thread (%d,%d), step %d] A=%.3f mu=%.3f sigma=%.3f v=%.3f c=%.3f x=%.3f y=%.3f\n",
  //     blockIdx.x, blockIdx.y,
  //     threadIdx.x, threadIdx.y,
  //     timestep,
  //     A_val, mu_val, sigma_val, vel, map_cost, s[0], s[1]
  //   );
  // }
  if (cost > MAX_COST_VALUE || isnan(cost))
    cost = MAX_COST_VALUE;
  return cost;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(float* s, float* theta_c)
{
  return 0.0f;
}

// ----------------------
// Cost Map Management
// ----------------------
template <class CLASS_T, class DYN_T, class PARAMS_T>
void DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::updateCostMapCPU(const std::vector<float4>& cost_map, int width, int height)
{
  if (width <= 0 || height <= 0) {
    std::cerr << "Error: invalid cost map dimensions." << std::endl;
    return;
  }
  if (width != map_width_ || height != map_height_ || costmapArray_d_ == nullptr) {
    if (costmapArray_d_ != nullptr) {
      cudaFreeArray(costmapArray_d_);
      costmapArray_d_ = nullptr;
    }
    map_width_ = width;
    map_height_ = height;
    cost_map_cpu_ = cost_map;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    HANDLE_ERROR(cudaMallocArray(&costmapArray_d_, &channelDesc, map_width_, map_height_));
  }
  else {
    cost_map_cpu_ = cost_map;
  }
  costMapToTexture();
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
void DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::costMapToTexture()
{
  if (map_width_ <= 0 || map_height_ <= 0) {
    std::cerr << "Error: cost map dimensions not set." << std::endl;
    return;
  }
  HANDLE_ERROR(cudaMemcpyToArray(costmapArray_d_, 0, 0, cost_map_cpu_.data(), map_width_ * map_height_ * sizeof(float4),
                                 cudaMemcpyHostToDevice));
  cudaStreamSynchronize(this->stream_);

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = costmapArray_d_;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  if (costmap_tex_d_ != 0) {
    HANDLE_ERROR(cudaDestroyTextureObject(costmap_tex_d_));
  }
  HANDLE_ERROR(cudaCreateTextureObject(&costmap_tex_d_, &resDesc, &texDesc, nullptr));

  HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->costmapArray_d_, &costmapArray_d_, sizeof(cudaArray*),
                                 cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->costmap_tex_d_, &costmap_tex_d_, sizeof(cudaTextureObject_t),
                                 cudaMemcpyHostToDevice, this->stream_));
  cudaStreamSynchronize(this->stream_);
}

// ----------------------
// Parameter Update Functions
// ----------------------
template <class CLASS_T, class DYN_T, class PARAMS_T>
void DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::paramsToDevice()
{
  HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->params_, &this->params_, sizeof(PARAMS_T),
                                 cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->map_width_, &map_width_, sizeof(int),
                                 cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->map_height_, &map_height_, sizeof(int),
                                 cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
}


// template <class CLASS_T, class DYN_T, class PARAMS_T>
// void DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::updateTransform(Eigen::MatrixXf m, Eigen::ArrayXf trs)
// {
//   // In this simplified design we no longer use a full projective transform.
//   // The transformation is computed in the node callback (see below).
//   // Here, we simply update the parameters if needed.
//   this->params_.r_c1.x = m(0, 0);
//   this->params_.r_c1.y = m(1, 0);
//   this->params_.r_c1.z = m(2, 0);
//   this->params_.r_c2.x = m(0, 1);
//   this->params_.r_c2.y = m(1, 1);
//   this->params_.r_c2.z = m(2, 1);
//   this->params_.trs.x  = trs(0);
//   this->params_.trs.y  = trs(1);
//   this->params_.trs.z  = trs(2);
//   if (this->GPUMemStatus_) {
//     paramsToDevice();
//   }
// }
