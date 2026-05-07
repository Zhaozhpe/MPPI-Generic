#include "AccelCurvatureDynamics.cuh"
#include <cmath>

// ctor: call real base class
AccelCurvatureDynamics::AccelCurvatureDynamics(cudaStream_t stream)
  : Dynamics<AccelCurvatureDynamics,AccelCurvatureParams>(stream)
{
  this->params_ = AccelCurvatureParams();
}

void AccelCurvatureDynamics::computeDynamics(
    const Eigen::Ref<const state_array>&  x,
    const Eigen::Ref<const control_array>& u,
          Eigen::Ref<state_array>          xdot)
{
  float v     = x[S_INDEX(VEL)];
  float kappa = x[S_INDEX(KAPPA)];

  xdot[S_INDEX(POS_X)] = v * std::cos(x[S_INDEX(YAW)]);
  xdot[S_INDEX(POS_Y)] = v * std::sin(x[S_INDEX(YAW)]);
  xdot[S_INDEX(YAW)]   = v * kappa;
  xdot[S_INDEX(VEL)]   = u[C_INDEX(ACCEL)];
  xdot[S_INDEX(KAPPA)] = u[C_INDEX(KAPPA_RATE)];
}

// void AccelCurvatureDynamics::updateState(
//     const Eigen::Ref<const state_array> state,
//           Eigen::Ref<state_array>      next_state,
//           Eigen::Ref<state_array>      state_der,
//     const float                      dt)
// {
//   // simple Euler + yaw‐normalization (no manual clamp)
//   next_state = state + state_der * dt;
//   next_state(S_INDEX(YAW)) = angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));
// }

void AccelCurvatureDynamics::updateState(
    const Eigen::Ref<const state_array> state,
          Eigen::Ref<state_array>      next_state,
          Eigen::Ref<state_array>      state_der,
    const float                      dt)
{
  // 1) Euler integration
  next_state = state + state_der * dt;

  // 2) normalize yaw
  next_state(S_INDEX(YAW)) =
    angle_utils::normalizeAngle(next_state(S_INDEX(YAW)));

  // 3) clamp linear speed
  float v_clamped = next_state(S_INDEX(VEL));
  v_clamped = std::min(std::max(v_clamped, VEL_MIN), VEL_MAX);
  next_state(S_INDEX(VEL)) = v_clamped;

  // 4) clamp curvature
  float k_clamped = next_state(S_INDEX(KAPPA));
  k_clamped = std::min(std::max(k_clamped, KAPPA_MIN), KAPPA_MAX);

  // 5) enforce ω = v·κ cap
  float omega = v_clamped * k_clamped;
  omega = std::min(std::max(omega, -OMEGA_MAX), OMEGA_MAX);

  // 6) back out κ so that κ·v = ω
  k_clamped = omega / (v_clamped + 1e-6f);
  next_state(S_INDEX(KAPPA)) = k_clamped;
}

__device__ void AccelCurvatureDynamics::computeDynamics(
    float* state, float* control, float* state_der, float* /*theta*/)
{
  float v     = state[S_INDEX(VEL)];
  float kappa = state[S_INDEX(KAPPA)];

  state_der[S_INDEX(POS_X)] = v * cosf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(POS_Y)] = v * sinf(state[S_INDEX(YAW)]);
  state_der[S_INDEX(YAW)]   = v * kappa;
  state_der[S_INDEX(VEL)]   = control[C_INDEX(ACCEL)];
  state_der[S_INDEX(KAPPA)] = control[C_INDEX(KAPPA_RATE)];
}

// __device__ void AccelCurvatureDynamics::updateState(
//     float* state, float* next_state, float* state_der, const float dt)
// {
//   int tdy = threadIdx.y;
//   for (int i = tdy; i < STATE_DIM; i += blockDim.y) {
//     next_state[i] = state[i] + state_der[i] * dt;
//     if (i == S_INDEX(YAW)) {
//       next_state[i] = angle_utils::normalizeAngle(next_state[i]);
//     }
//   }
// }

__device__ void AccelCurvatureDynamics::updateState(
    float* state, float* next_state, float* state_der, const float dt)
{
  for (int i = threadIdx.y; i < STATE_DIM; i += blockDim.y) {
    // 1) integrate
    float val = state[i] + state_der[i] * dt;

    // 2) yaw‐normalize
    if (i == S_INDEX(YAW)) {
      val = angle_utils::normalizeAngle(val);
    }
    // 3) clamp v
    else if (i == S_INDEX(VEL)) {
      val = fminf(fmaxf(val, VEL_MIN), VEL_MAX);
    }
    // 4) clamp κ and enforce ω cap
    else if (i == S_INDEX(KAPPA)) {
      // clamp curvature
      val = fminf(fmaxf(val, KAPPA_MIN), KAPPA_MAX);

      // compute what the new v will be
      float v_new = state[S_INDEX(VEL)] + state_der[S_INDEX(VEL)] * dt;
      v_new = fminf(fmaxf(v_new, VEL_MIN), VEL_MAX);

      // enforce ω ≤ Ωₘₐₓ
      float omega = v_new * val;
      omega = fminf(fmaxf(omega, -OMEGA_MAX), OMEGA_MAX);

      // back out κ from the clamped ω
      val = omega / (v_new + 1e-6f);
    }

    next_state[i] = val;
  }
}


Dynamics<AccelCurvatureDynamics,AccelCurvatureParams>::state_array
AccelCurvatureDynamics::stateFromMap(const std::map<std::string, float>& map)
{
  state_array s;
  s(S_INDEX(POS_X)) = map.at("POS_X");
  s(S_INDEX(POS_Y)) = map.at("POS_Y");
  s(S_INDEX(YAW))   = map.at("YAW");
  s(S_INDEX(VEL))   = map.at("VEL");
  s(S_INDEX(KAPPA)) = map.at("KAPPA");
  return s;
}
