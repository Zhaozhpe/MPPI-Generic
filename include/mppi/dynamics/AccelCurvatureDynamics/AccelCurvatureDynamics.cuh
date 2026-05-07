#pragma once

#ifndef ACCEL_CURVATURE_DYNAMICS_CUH_
#define ACCEL_CURVATURE_DYNAMICS_CUH_

#include <mppi/dynamics/dynamics.cuh>
#include <mppi/utils/angle_utils.cuh>
#include <random>
#include <algorithm>

#define VEL_MIN   0.0f
#define VEL_MAX  20.0f   // e.g. 20 m/s max
#define KAPPA_MIN -1.0f  // e.g. max curvature
#define KAPPA_MAX  1.0f
#define OMEGA_MAX  3.14f

struct AccelCurvatureParams : public DynamicsParams
{
  enum class StateIndex : int
  {
    POS_X = 0,
    POS_Y,
    YAW,
    VEL,
    KAPPA,
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    ACCEL = 0,
    KAPPA_RATE,
    NUM_CONTROLS
  };

  enum class OutputIndex : int
  {
    POS_X = 0,
    POS_Y,
    YAW,
    VEL,
    KAPPA,
    NUM_OUTPUTS
  };

  AccelCurvatureParams() = default;
  ~AccelCurvatureParams() = default;
};

using namespace MPPI_internal;

/**
 * state:  [x, y, yaw, v, κ]
 * control: [a, κ̇]
 */
class AccelCurvatureDynamics : public Dynamics<AccelCurvatureDynamics, AccelCurvatureParams>
{
public:
  AccelCurvatureDynamics(cudaStream_t stream = nullptr);

  using PARENT_CLASS = Dynamics<AccelCurvatureDynamics, AccelCurvatureParams>;
  using PARENT_CLASS::updateState;  // bring in base overloads

  std::string getDynamicsModelName() const override
  {
    return "Accel+Curvature Model";
  }

  // CPU‐side
  void computeDynamics(
    const Eigen::Ref<const state_array>&  state,
    const Eigen::Ref<const control_array>& control,
          Eigen::Ref<state_array>          state_der);

  void updateState(
    const Eigen::Ref<const state_array> state,
          Eigen::Ref<state_array>       next_state,
          Eigen::Ref<state_array>       state_der,
    const float                       dt);

  // GPU‐side
  __device__ void computeDynamics(
    float* state, float* control, float* state_der, float* /*theta*/);

  __device__ void updateState(
    float* state, float* next_state, float* state_der, const float dt);

  // allow constructing a state from a map if you need it
  state_array stateFromMap(const std::map<std::string, float>& map) override;

private:
  // float min_vel_, max_vel_, max_kappa_;
};

#if __CUDACC__
#include "AccelCurvatureDynamics.cu"
#endif

#endif  // ACCEL_CURVATURE_DYNAMICS_CUH_
