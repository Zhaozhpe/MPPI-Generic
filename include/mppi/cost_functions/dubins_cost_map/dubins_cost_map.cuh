#pragma once
#ifndef MPPI_COST_FUNCTIONS_DUBINS_COST_MAP_CUH_
#define MPPI_COST_FUNCTIONS_DUBINS_COST_MAP_CUH_

#include <mppi/cost_functions/cost.cuh>
#include <mppi/dynamics/dubins/dubins.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>
#include <cstring>  // for memset
#include <cmath>

// Parameters for DubinsCostMap.
template <class DYN_T, int SIM_TIME_HORIZON = 1>
struct DubinsCostMapParams : public CostParams<DYN_T::CONTROL_DIM>
{
  // Desired state for quadratic tracking (size = OUTPUT_DIM * SIM_TIME_HORIZON)
  float s_goal[DYN_T::OUTPUT_DIM * SIM_TIME_HORIZON] = { 0 };
  
  // state cost coefficients
  float s_coeffs[DYN_T::OUTPUT_DIM] = { 0 }; // Coefficients for quadratic state error.
  // map cost coefficients
  float map_cost_weight[3] = {30.0f, 500.0f, 1000.0f}; // Weight for the cost-map term.
  float terminal_cost_weight = 100.0f;

  int current_time = 0;
  int timesteps_total = 100;
  bool adaptive_cost = true;

  // New fields: vehicle state (center of grid) and region size (in meters).
  float vehicle_x = 0.0f;
  float vehicle_y = 0.0f;
  float vehicle_yaw = 0.0f;
  float region_size = 40.0f; // e.g. 40 meters square

  DubinsCostMapParams()
  {
    for (int i = 0; i < DYN_T::CONTROL_DIM; i++) {
      this->control_cost_coeff[i] = 0.1f;
    }
    s_coeffs[0] = 5.0f;
    s_coeffs[1] = 5.0f; 
    s_coeffs[2] = 2.0f; 
  }
  
  // Host function
  // Returns the desired state at timestep t. 
  const Eigen::Matrix<float, DYN_T::OUTPUT_DIM, 1> getDesiredState(int t)
  {
    Eigen::Matrix<float, DYN_T::OUTPUT_DIM, 1> s(s_goal + getIndex(t));
    return s;
  }

  __device__ float* getGoalStatePointer()
  {
    return s_goal;
  }

  __host__ __device__ int getIndex(int t)
  {
    int index = current_time + t;
    if (index >= SIM_TIME_HORIZON) {
      index = SIM_TIME_HORIZON - 1;
    }
    index *= DYN_T::OUTPUT_DIM;
    return index;
  }

  __host__ __device__ void setCurrentTime(int new_time)
  {
    current_time = new_time;
  }
};

template <class CLASS_T, class DYN_T, class PARAMS_T>
class DubinsCostMapImpl : public Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T>
{
public:
  typedef Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T> PARENT_CLASS;
  using output_array = typename PARENT_CLASS::output_array;
  static constexpr float MAX_COST_VALUE = 1e16;

  DubinsCostMapImpl(cudaStream_t stream = nullptr);

  virtual std::string getCostFunctionName() const override { return "Dubins Cost Map"; }

  // Host functions: compute quadratic state error plus map cost.
  float computeStateCost(const Eigen::Ref<const output_array> s, int timestep = 0, int* crash_status = nullptr);
  float terminalCost(const Eigen::Ref<const output_array> s);

  // Device functions.
  __device__ float computeStateCost(float* s, float* u, int timestep = 0, float* theta_c = nullptr, int* crash_status = nullptr);
  __device__ float terminalCost(float* s, float* theta_c);

  // Methods to update the cost map.
  void updateCostMapCPU(const std::vector<float4>& cost_map, int width, int height);
  void costMapToTexture();

  // Setter for the desired state (avoids direct access to params_).
  void setDesiredState(const Eigen::Vector3f& desired_state)
  {
      this->params_.s_goal[0] = desired_state(0);
      this->params_.s_goal[1] = desired_state(1);
      this->params_.s_goal[2] = desired_state(2);
  }

  void setVehicleState(float x, float y, float yaw)
  {
    this->params_.vehicle_x = x;
    this->params_.vehicle_y = y;
    this->params_.vehicle_yaw = yaw;
  }

  void setRegionSize(float region_size_val)
  {
    this->params_.region_size = region_size_val;
  }
  void setTimestepsTotal(int t)
  {
    this->params_.timesteps_total = t;
  }

  void setStateCostWeights(const std::array<float, 5>& w) {
    for (int i = 0; i < 5; i++)
      this->params_.s_coeffs[i] = w[i];
  }

  void serTerminalCostWeight(float w) {
    this->params_.terminal_cost_weight = w;
  }

  /// Set the per‐control quadratic weights [w_vel, w_yaw_rate]
  void setControlCostWeights(const std::array<float, 2>& w) {
    for (int i = 0; i < 2; i++)
      this->params_.control_cost_coeff[i] = w[i];
  }

  void setMapCostWeight(const std::array<float, 3>& w) {
    for (int i = 0; i < 3; i++)
      this->params_.map_cost_weight[i] = w[i];
  }

  void setAdaptiveCost(bool adaptive_cost) {
    this->params_.adaptive_cost = adaptive_cost;
  }

  // void setAccelCostWeight(float w) {
  //   this->params_.accel_cost_coeff = w;
  // }

  // Copies updated parameters to the GPU.
  void paramsToDevice();

  // Update the cost map transformation using Eigen matrices,
  // similar to ARStandardCost. (This assumes a 3x3 transform matrix and a 3-element translation.)
  void updateTransform(Eigen::MatrixXf m, Eigen::ArrayXf trs);
  __device__ float computeRunningCost(float* y, float* u,
                                      int timestep, float* theta_c, int* crash);

protected:
  // Cost map dimensions and GPU data.
  int map_width_ = -1;
  int map_height_ = -1;
  cudaArray* costmapArray_d_ = nullptr;
  cudaTextureObject_t costmap_tex_d_ = 0;
  std::vector<float4> cost_map_cpu_;

  // Coordinate transformation: converts (x,y) in world frame to (u,v,w) for texture lookup.
  __host__ __device__ void coorTransform(float x, float y, float* u, float* v) const
  {
    // x, y are query points in the world frame.
    // Subtract vehicle state to get relative coordinates.
    float dx = x - this->params_.vehicle_x;
    float dy = y - this->params_.vehicle_y;
    // Rotate into vehicle frame (assumes vehicle_yaw is the heading).
    float theta = this->params_.vehicle_yaw;
    float local_x = cosf(theta) * dx + sinf(theta) * dy;
    float local_y = -sinf(theta) * dx + cosf(theta) * dy;
    // Now, local coordinates should lie in [-region_size/2, region_size/2].
    // Map these to normalized coordinates in [0,1]:
    *u = (local_x + this->params_.region_size / 2.0f) / this->params_.region_size;
    *v = (local_y + this->params_.region_size / 2.0f) / this->params_.region_size;
  }

  // Device-only function to query the cost map texture.
  __device__ float queryTexture(float x, float y) const
  {
    return tex2D<float4>(costmap_tex_d_, x, y).x;
  }

  // Query that transforms (x,y) and then performs texture lookup.
  __host__ __device__ float4 queryTextureTransformed(float x, float y) const
  {
    float u, v;
    coorTransform(x, y, &u, &v);
#ifdef __CUDA_ARCH__
    // printf("queryTextureTransformed (device): x=%f, y=%f, u=%f, v=%f\n", x, y, u, v);
    float4 tex_val = tex2D<float4>(costmap_tex_d_, u, v);
    // float norm_u = (139.0f + 0.5f) / (float)map_width_;
    // float norm_v = (135.0f + 0.5f) / (float)map_height_;
    // float tex_val = tex2D<float4>(costmap_tex_d_, norm_u, norm_v).x;
    // if (tex_val > 0.0f) {
    //   // printf("tex2D returned: %f\n", tex_val);
    // }
    return tex_val;
    // return tex2D<float4>(costmap_tex_d_, u, v).x;
#else
    // On the host, convert normalized coordinates to pixel indices.
    float query_x = u * (float)map_width_;
    float query_y = v * (float)map_height_;
    query_x = fmaxf(0.0f, fminf(float(map_width_ - 1), query_x));
    query_y = fmaxf(0.0f, fminf(float(map_height_ - 1), query_y));
    int idx = std::round(query_y) * map_width_ + std::round(query_x);
    return cost_map_cpu_[idx];
#endif
  }
};

#if __CUDACC__
#include "dubins_cost_map.cu"
#endif

// Convenience typedefs.
template <class DYN_T, int SIM_TIME_HORIZON = 1>
class DubinsCostMapTrajectory : public DubinsCostMapImpl<DubinsCostMapTrajectory<DYN_T, SIM_TIME_HORIZON>, DYN_T, DubinsCostMapParams<DYN_T, SIM_TIME_HORIZON>>
{
public:
  DubinsCostMapTrajectory(cudaStream_t stream = nullptr)
    : DubinsCostMapImpl<DubinsCostMapTrajectory, DYN_T, DubinsCostMapParams<DYN_T, SIM_TIME_HORIZON>>(stream) {}
};

template <class DYN_T>
class DubinsCostMap : public DubinsCostMapImpl<DubinsCostMap<DYN_T>, DYN_T, DubinsCostMapParams<DYN_T>>
{
public:
  DubinsCostMap(cudaStream_t stream = nullptr)
    : DubinsCostMapImpl<DubinsCostMap, DYN_T, DubinsCostMapParams<DYN_T>>(stream) {}
};

#endif // MPPI_COST_FUNCTIONS_DUBINS_COST_MAP_CUH_
