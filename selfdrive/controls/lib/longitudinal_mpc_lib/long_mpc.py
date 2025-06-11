#!/usr/bin/env python3
import os
import time
import numpy as np
from cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.toyota.values import ToyotaFlags
from openpilot.common.conversions import Conversions as CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

if __name__ == '__main__':  # generating code
  from openpilot.third_party.acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

SOURCES = ['lead0', 'lead1', 'cruise', 'e2e']

X_DIM = 3
U_DIM = 1
PARAM_DIM = 7
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

#安全 vs 舒適：
  #提高 X_EGO_OBSTACLE_COST 和 DANGER_ZONE_COST → 優先安全，保持較大跟車距離
  #提高 A_EGO_COST、A_CHANGE_COST、J_EGO_COST → 優先平順，減少急加減速與指令突變

#反應速度 vs 保守性：
  #降低 V_EGO_COST → 願意跑更高車速，提高追趕或切入的積極度
  #降低 X_EGO_COST → 願意往前移動，準備加速跟上前方車流
X_EGO_OBSTACLE_COST = 2. # 降低避障成本以避免過於保守
X_EGO_COST = 1.0  # 增加以提升車距追蹤精度
V_EGO_COST = 1.0  # 適度權重於自車速度
A_EGO_COST = 5.0  # 對加速度施加小懲罰以平滑動作曲線
J_EGO_COST = 3.0  # 降低 jerk 懲罰以提高反應靈敏度
A_CHANGE_COST = 175.  # 降低以提供更大加速自由度
#DANGER_ZONE_COST = 350. #100.
CRASH_DISTANCE = .25
#LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
NUMERIC_EPS = 1e-4  # 小數值以避免除以零或數值不穩定
ACADOS_SOLVER_TYPE = 'SQP_RTI'


# 減少時間點不會影響效能並能帶來
# 更好的 MPC 收斂效果，且所需疊代次數更少
N = 16 #12
MAX_T = 15.0 #10.0
# 根據 N 與 MAX_T 調整的預測時間範圍
#T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]
#T_IDXS = np.array(T_IDXS_LST)
T_IDXS = (np.linspace(0, 1, N + 1) ** 2.0) * MAX_T # 調整 **數字提升前其靈敏度(2.0前段密集、後段拉開明顯, 2.5-3.0前段極度靈敏（不自然）)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
# STOP_DISTANCE = 6.0
CRUISE_MIN_ACCEL = -1.2
CRUISE_MAX_ACCEL = 1.6

def get_danger_zone_cost(v_ego):
  # 線性插值：0 m/s → 100，33.3 m/s (120 km/h) → 300
  return np.interp(v_ego, [0.0, 16.67], [100.0, 450.0])

def get_lead_danger_factor(v_ego):
  # 線性插值：0 m/s → 1.0，33.3 m/s (120 km/h) → 1.5
  return np.interp(v_ego, [0.0, 33.3], [1.1, 1.6])

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 1.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.3
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.35
  elif personality==log.LongitudinalPersonality.standard:
    return 1.25
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.0
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_dynamic_follow(v_ego, personality=log.LongitudinalPersonality.standard):
  # The Dynamic follow function is adjusted by Marc(cgw1968-5779)
  if personality==log.LongitudinalPersonality.relaxed:
    x_vel =  [0.,  6,   10., 10.01, 15., 27.7]
    y_dist = [1.2, 1.4, 1.4,  1.5, 1.65,  1.8]
  elif personality==log.LongitudinalPersonality.standard:
    x_vel =  [0.,  6,   10., 10.01, 15., 27.7]
    y_dist = [1.1, 1.3, 1.35, 1.4,  1.4, 1.45]
  elif personality==log.LongitudinalPersonality.aggressive:
    x_vel =  [0.,  6,   10., 10.01, 15., 27.7]
    y_dist = [1.0, 1.2, 1.0,   0.9, 0.95, 1.0]
  else:
    raise NotImplementedError("Dynamic Follow personality not supported")
  return np.interp(v_ego, x_vel, y_dist)


def get_STOP_DISTANCE(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 6.0
  elif personality==log.LongitudinalPersonality.standard:
    return 6.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 6.0
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_stopped_equivalence_factor(v_lead, v_ego):
  # KRKeegan this offset rapidly decreases the following distance when the lead pulls
  # away, resulting in an early demand for acceleration.
  v_diff_offset = 0
  v_diff_offset_max = 5 #12
  speed_to_reach_max_v_diff_offset = 12 #26 # in kp/h
  speed_to_reach_max_v_diff_offset = speed_to_reach_max_v_diff_offset * CV.KPH_TO_MS
  delta_speed = v_lead - v_ego
  if np.all(delta_speed > 0):
    v_diff_offset = (np.clip(delta_speed, 0, 5)) ** 2.5
    v_diff_offset = np.clip(v_diff_offset, 0, v_diff_offset_max)
    v_diff_offset = np.maximum(v_diff_offset * ((speed_to_reach_max_v_diff_offset - v_ego)/speed_to_reach_max_v_diff_offset), 0)
  return (v_lead**2) / (2 * COMFORT_BRAKE) + v_diff_offset

def get_safe_obstacle_distance(v_ego, t_follow, stop_distance=None, personality=log.LongitudinalPersonality.standard):
  if stop_distance is None:
    stop_distance = get_STOP_DISTANCE(personality)
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + stop_distance

def desired_follow_distance(v_ego, v_lead, t_follow=None, stop_distance=None, personality=log.LongitudinalPersonality.standard):
  if t_follow is None:
    t_follow = get_T_FOLLOW(personality)
  if stop_distance is None:
    stop_distance = get_STOP_DISTANCE(personality)
  return max(0.0, get_safe_obstacle_distance(v_ego, t_follow, stop_distance) - get_stopped_equivalence_factor(v_lead, v_ego))


def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # live parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  prev_a = SX.sym('prev_a')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  stop_distance = SX.sym('stop_distance')
  model.p = vertcat(a_min, a_max, x_obstacle, prev_a, lead_t_follow, lead_danger_factor, stop_distance)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  prev_a = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]
  stop_distance = ocp.model.p[6]

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow, stop_distance)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),
           x_ego,
           v_ego,
           a_ego,
           a_ego - prev_a,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego,
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), get_lead_danger_factor(v_ego), get_STOP_DISTANCE()])


  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, CP, mode='acc', dt=DT_MDL):
    self.CP = CP
    self.mode = mode
    self.dt = dt
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.reset()
    self.source = SOURCES[2]

  def reset(self):
    # self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.solver.reset()
    # self.solver.options_set('print_level', 2)
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.prev_a = np.array(self.a_solution)
    self.j_solution = np.zeros(N)
    self.yref = np.zeros((N+1, COST_DIM))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N,1))
    self.params = np.zeros((N+1, PARAM_DIM))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
    self.last_cloudlog_t = 0
    self.status = False
    self.crash_cnt = 0.0
    self.solution_status = 0
    # timers
    self.solve_time = 0.0
    self.time_qp_solution = 0.0
    self.time_linearization = 0.0
    self.time_integrator = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard, v_lead0=0, v_lead1=0):
    jerk_factor = get_jerk_factor(personality)
    v_ego = self.x0[1]
    v_ego_bps = [0, 10]
    # KRKeegan adjustments to improve sluggish acceleration
    # do not apply to deceleration
    j_ego_v_ego = 1
    a_change_v_ego = 1
    if (v_lead0 - v_ego >= 0) and (v_lead1 - v_ego >= 0):
      j_ego_v_ego = np.interp(v_ego, v_ego_bps, [.10, 1.])
      a_change_v_ego = np.interp(v_ego, v_ego_bps, [.10, 1.])

    danger_cost = get_danger_zone_cost(v_ego)
    
    if self.mode == 'acc':
      a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
      cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost * a_change_v_ego, jerk_factor * J_EGO_COST * j_ego_v_ego]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, danger_cost]
    elif self.mode == 'blended':
      a_change_cost = 150.0 if prev_accel_constraint else 0
      #cost_weights = [2., 1.0, 1.0, 5.0, a_change_cost * a_change_v_ego, 1.0]
      cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost * a_change_v_ego, jerk_factor * J_EGO_COST * j_ego_v_ego]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, danger_cost]
    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner cost set')
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      for i in range(N+1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    #a_lead_traj = a_lead * np.exp(-T_DIFFS.cumsum() / a_lead_tau)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def update(self, radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard, dynamic_follow=False):
    t_follow = get_T_FOLLOW(personality)
    v_ego = self.x0[1]
    speed_kph = v_ego * 3.6
    # 1. 動態算出時間節點
    #T_IDXS_loc = compute_T_IDXS(v_ego)
    #T_DIFFS_loc = np.diff(T_IDXS, prepend=[0.])
    t_follow = get_T_FOLLOW(personality) if not dynamic_follow else get_dynamic_follow(v_ego, personality)
    stop_distance = get_STOP_DISTANCE(personality)

    if Params().get_bool("ToyotaTune") and not (self.CP.flags & ToyotaFlags.SMART_DSU):
      stop_distance += 3.0

    self.status = radarstate.leadOne.status or radarstate.leadTwo.status

    lead_xv_0 = self.process_lead(radarstate.leadOne)
    lead_xv_1 = self.process_lead(radarstate.leadTwo)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1], v_ego)
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1], v_ego)

    self.params[:,0] = ACCEL_MIN
    self.params[:,1] = ACCEL_MAX

    # Update in ACC mode or ACC/e2e blend
    if self.mode == 'acc':
      self.params[:,5] = get_lead_danger_factor(v_ego)

      # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
      # when the leads are no factor.
      v_lower = v_ego + (T_IDXS * CRUISE_MIN_ACCEL * 0.95) # *越大,減速越保守
      # TODO does this make sense when max_a is negative?
      v_upper = v_ego + (T_IDXS * CRUISE_MAX_ACCEL * 0.9) # *越大,加速越激進
      v_cruise_clipped = np.clip(v_cruise * np.ones(N+1),
                                 v_lower,
                                 v_upper)
      cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow, stop_distance)
      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
      self.source = SOURCES[np.argmin(x_obstacles[0])]

      # These are not used in ACC mode
      x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

    elif self.mode == 'blended':
      self.params[:,5] = get_lead_danger_factor(v_ego) #1.0
      
      v_lower = v_ego + (T_IDXS * CRUISE_MIN_ACCEL * 0.95) # *越大,減速越保守
      # TODO does this make sense when max_a is negative?
      v_upper = v_ego + (T_IDXS * CRUISE_MAX_ACCEL * 0.9) # *越大,加速越激進
      v_cruise_clipped = np.clip(v_cruise * np.ones(N+1),
                                 v_lower,
                                 v_upper)
      cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow, stop_distance)
      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
      #x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle])
      # cruise 目標距離
      cruise_target = T_IDXS * np.clip(v_cruise, v_ego - 2.0, 1e3) + x[0] # *1.0是放大係數（可改為 >1.0 讓巡航更激進，或 <1.0 更保守），下限 v_ego - 2.0 決定了當車速高於目標時是否允許輕微減速。
      # —— 1) 動態縮減巡航速度 ——
      #if speed_kph > 90:
        #scale = np.interp(speed_kph, [90, 120], [1.0, 0.7])
      #else:
        #scale = 1
        
      #adj_v_cruise = v_cruise * scale

      # —— 2) 純巡航軌跡 + 安全距離限制 ——
      #cruise_base = T_IDXS * np.clip(v_cruise, v_ego - 2.0, 1e3) + x[0]
      # 計算安全跟車距
      #safe_dist = desired_follow_distance(v_ego, v_lead, t_follow)
      #lead_pos0 = np.min([lead_0_obstacle[0], lead_1_obstacle[0]])
      #cruise_target = np.minimum(cruise_base, lead_pos0 - safe_dist)
      #=======================================================================
      # e2e 預測距離
      xforward = ((v[1:] + v[:-1]) / 2) * (T_IDXS[1:] - T_IDXS[:-1]) 
      x = np.cumsum(np.insert(xforward, 0, x[0]))

      # 混合 e2e 和 cruise，根據速度平滑插值
      x_and_cruise = np.column_stack([x * 1.0, cruise_target]) # 將 e2e 預測距離額外乘以 0.95，會讓 e2e 軌跡對加速目標略顯保守。數值越接近 1，e2e 的影響越大；越小，則更偏向 cruise，進而影響加速決策和引擎轉速。
      #x = np.max(x_and_cruise, axis=1)
      #計算速度加權：低速偏 e2e，高速偏 cruise
      w = np.clip((speed_kph - 20.0) / 100.0, 0.0, 0.3)  #15
      # 高速時再衰減
      #decay = np.interp(speed_kph, [100, 140], [1.0, 0.5])
      #w *= decay
      x = (1 - w) * np.min(x_and_cruise, axis=1) + w * np.max(x_and_cruise, axis=1)
      #==========================================================================
      # 若 e2e 比 cruise 明顯遠，才使用 e2e 作為來源
      if speed_kph < 60:
        if x_and_cruise[1,0] > 1.1 * x_and_cruise[1,1]:
          self.source = 'e2e'
        else:
          self.source = 'cruise'
      else:
        self.source = 'cruise'

      #if speed_kph < 60:
        #self.source = 'e2e'
      #else:
        #self.source = 'cruise'

    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner update')

    self.yref[:,1] = x
    self.yref[:,2] = v
    self.yref[:,3] = a
    self.yref[:,5] = j
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.prev_a)
    self.params[:,4] = t_follow
    self.params[:,6] = stop_distance

    self.run()
    if (np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

    # Check if it got within lead comfort range
    # TODO This should be done cleaner
    if self.mode == 'blended':
      if any((lead_0_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow, stop_distance))- self.x_sol[:,0] < 0.0):
        self.source = 'lead0'
      if any((lead_1_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow, stop_distance))- self.x_sol[:,0] < 0.0) and \
         (lead_1_obstacle[0] > lead_0_obstacle[0]):
        self.source = 'lead1'

  def run(self):
    # t0 = time.monotonic()
    # reset = 0
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])
    self.time_qp_solution = float(self.solver.get_stats('time_qp')[0])
    self.time_linearization = float(self.solver.get_stats('time_lin')[0])
    self.time_integrator = float(self.solver.get_stats('time_sim')[0])

    # qp_iter = self.solver.get_stats('statistics')[-1][-1] # SQP_RTI specific
    # print(f"long_mpc timings: tot {self.solve_time:.2e}, qp {self.time_qp_solution:.2e}, lin {self.time_linearization:.2e}, \
    # integrator {self.time_integrator:.2e}, qp_iter {qp_iter}")
    # res = self.solver.get_residuals()
    # print(f"long_mpc residuals: {res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}, {res[3]:.2e}")
    # self.solver.print_statistics()

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    self.prev_a = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    t = time.monotonic()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()
      # reset = 1
    # print(f"long_mpc timings: total internal {self.solve_time:.2e}, external: {(time.monotonic() - t0):.2e} qp {self.time_qp_solution:.2e}, \
    # lin {self.time_linearization:.2e} qp_iter {qp_iter}, reset {reset}")


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
