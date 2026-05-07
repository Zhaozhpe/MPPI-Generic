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
  return 888.0f;
}

// ----------------------
// Device Functions
// ----------------------
template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float DubinsCostMapImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(float* s, int timestep, float* theta_c, int* crash_status)
{
  // query timestep: 0 to 99
  float state_cost = 0.0f;
  float* desired_state = this->params_.getGoalStatePointer();

  float distance_error = 0.0f;

  if (timestep >= (this->params_.timesteps_total - 10)) {
    float dx = s[0] - desired_state[0];
    float dy = s[1] - desired_state[1];
    distance_error = dx * dx * this->params_.s_coeffs[0] + dy * dy * this->params_.s_coeffs[1];
  }

  // Yaw error only considered at final timestep
  float yaw_error = 0.0f;
  if (timestep == (this->params_.timesteps_total - 1)) {
      float dyaw = s[2] - desired_state[2];
      // Normalize yaw error between [-pi, pi] if necessary
      dyaw = atan2f(sinf(dyaw), cosf(dyaw));
      yaw_error = dyaw * dyaw * this->params_.s_coeffs[2]; 
  }
  state_cost = distance_error + yaw_error;

  // for (int i = 0; i < DYN_T::OUTPUT_DIM; i++) {
  //   float error = s[i] - desired_state[i];
  //   state_cost += error * error * this->params_.s_coeffs[i];
  // }
  float map_cost = queryTextureTransformed(s[0], s[1]);
  float obs_cost = this->params_.map_cost_weight * fabsf(map_cost);
  float crash_penalty = 0.0f;
  if (fabsf(map_cost) > 0.5f) {
    crash_penalty = 1000.0f;
    if (crash_status != nullptr)
      crash_status[0] = 1;
  }
  float cost = state_cost + obs_cost + crash_penalty; 
  
  // float cost = 10000.0f* timestep; 
  // printf("computeStateCost: timestep = %d, state = [%f, %f, %f], desired_state = [%f, %f, %f], state_cost = %f, obs_cost = %f, crash_cost = %f\n",
    // timestep, s[0], s[1], s[2], desired_state[0], desired_state[1], desired_state[2], state_cost, obs_cost, crash_penalty);
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
