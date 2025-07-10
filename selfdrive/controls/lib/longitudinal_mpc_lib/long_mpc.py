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

X_EGO_OBSTACLE_COST = 2. # 降低避障成本以避免過於保守
X_EGO_COST = 0.  # 增加以提升車距追蹤精度
V_EGO_COST = 0.  # 適度權重於自車速度
A_EGO_COST = 0.  # 對加速度施加小懲罰以平滑動作曲線
J_EGO_COST = 5.0  # 降低 jerk 懲罰以提高反應靈敏度
A_CHANGE_COST = 100.  # 降低以提供更大加速自由度
#DANGER_ZONE_COST = 300.
CRASH_DISTANCE = .25
#LEAD_DANGER_FACTOR = 0.85 #0.75
LIMIT_COST = 1e6
NUMERIC_EPS = 1e-4  # 小數值以避免除以零或數值不穩定
ACADOS_SOLVER_TYPE = 'SQP_RTI'


# 減少時間點不會影響效能並能帶來
# 更好的 MPC 收斂效果，且所需疊代次數更少
N = 16 #12
MAX_T = 15.0 #10.0
# 根據 N 與 MAX_T 調整的預測時間範圍
#T_IDXS_LST = np.linspace(0, MAX_T, N + 1) ** 1.2  # 強化短期預測的精度
#T_IDXS = np.array(T_IDXS_LST)
T_IDXS = (np.linspace(0, 1, N + 1) ** 2.0) * MAX_T # 調整 **數字提升前其靈敏度(2.0前段密集、後段拉開明顯, 2.5-3.0前段極度靈敏（不自然）)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
# STOP_DISTANCE = 6.0
CRUISE_MIN_ACCEL = -1.2
CRUISE_MAX_ACCEL = 1.6

#===================================================================
# 閾值（m/s）
low_thr  = 20.0 / 3.6   # km/hr to m/s
mid_thr = 30.0 / 3.6   # km/hr to m/s
high_thr = 70.0 / 3.6
#===================================================================


def get_danger_zone_cost(v_ego):
  # 線性插值：0 m/s → 100，33.3 m/s (120 km/h) → 300
  #return np.interp(v_ego, [0.0, 27.78], [120.0, 500.0])
  if v_ego <= mid_thr:
    return 350.0
  elif v_ego <= high_thr:
    return 300.0#np.interp(v_ego, [10.0, 19.5], [130.0, 300.0])
  else:
    return 350.0#np.interp(v_ego, [19.5, 27.8], [300.0, 600.0])

#def get_lead_danger_factor(v_ego):
  #return np.interp(v_ego, [0.0, 33.3], [1.0, 1.4])  # 線性插值，隨速度提升危險因子增加

def get_lead_danger_factor(v_ego):
  if v_ego <= mid_thr:
    return 0.9
  elif v_ego <= high_thr:
    return 0.9
  else:
    return 1.0

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.5
  elif personality==log.LongitudinalPersonality.standard:
    return 1.3
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.3
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.3
  elif personality==log.LongitudinalPersonality.standard:
    return 1.2
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

def get_adaptive_T_FOLLOW(v_ego, a_lead, personality=log.LongitudinalPersonality.standard):
  # 基本 T_FOLLOW
  base_t_follow = get_T_FOLLOW(personality)

  # 當前車有明顯減速時，額外增加安全距離
  if a_lead < -3.0:
    # 增加最多0.3秒追車時距，視前車減速度線性調整
    extra_t_follow = np.clip(-0.3 * a_lead, 0.0, 0.3)
    base_t_follow += extra_t_follow

  return base_t_follow

def get_STOP_DISTANCE(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 5.0
  elif personality==log.LongitudinalPersonality.standard:
    return 5.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 5.0
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_stopped_equivalence_factor(v_lead, v_ego):
  # KRKeegan this offset rapidly decreases the following distance when the lead pulls
  # away, resulting in an early demand for acceleration.
  v_diff_offset = 0
  v_diff_offset_max = 2 #12,5
  speed_to_reach_max_v_diff_offset = 8 #26,12 # in kp/h
  speed_to_reach_max_v_diff_offset = speed_to_reach_max_v_diff_offset * CV.KPH_TO_MS
  delta_speed = v_lead - v_ego
  if np.all(delta_speed > 0.0):
    v_diff_offset = (np.clip(delta_speed, 0, 5)) ** 2.5
    v_diff_offset = np.clip(v_diff_offset, 0, v_diff_offset_max)
    v_diff_offset = np.maximum(v_diff_offset * ((speed_to_reach_max_v_diff_offset - v_ego)/speed_to_reach_max_v_diff_offset), 0)
  return (v_lead**2) / (2 * COMFORT_BRAKE) + v_diff_offset

def get_safe_obstacle_distance(v_ego, t_follow, stop_distance=None):
  if stop_distance is None:
    stop_distance = get_STOP_DISTANCE()
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + stop_distance

def desired_follow_distance(v_ego, v_lead, t_follow=None, stop_distance=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  if stop_distance is None:
    stop_distance = get_STOP_DISTANCE()
  return max(get_safe_obstacle_distance(v_ego, t_follow, stop_distance) - get_stopped_equivalence_factor(v_lead, v_ego), 4.0)


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
  #ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR, get_STOP_DISTANCE()])
  # 初始值固定為 1.0，運行時將由 get_lead_danger_factor(v_ego) 動態覆蓋
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), 1.0, get_STOP_DISTANCE()])


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
    self.params_store = Params()
    self.reset()
    self.source = SOURCES[2]

    self.prev_v_ego = 0.0 # 新增：用來記錄上一次的速度

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
      #W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 2.0, 4.0], [1.0, 0.5, 0.0])
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
    
    v_lead = v_lead0 if v_lead0 < v_lead1 else v_lead1
    relative_dist = np.clip(v_lead - v_ego, -5.0, 5.0)

    j_ego_v_ego = np.interp(v_ego, [0, mid_thr, high_thr], [0.3, 1.0, 2.0])       # 高速 jerk cost 高
    a_change_v_ego = np.interp(relative_dist, [-1.0, 0.0, 1.0], [1.2, 1.0, 0.7])  # 前車遠 → 提高靈敏度
    #========================
    danger_cost = get_danger_zone_cost(v_ego)
    #cost_weights = [跟車距離誤差,權重越大，MPC 越嚴格維持安全距離 / 絕對位置：對車輛位置的懲罰 / 速度跟蹤：對車速的懲罰 
                        #/ 加速度能量：對加速度本身的懲罰 / 加速度變化量（Δa）：懲罰連續兩步之間的加速度跳變 / jerk（控制輸入）：對加速度指令的變化率直接懲罰]
    if self.mode == 'acc':
      #danger_cost = 150.
      jerk_comf = 3.0
      if v_ego > high_thr:
        jerk_comf *= 3.0
      a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
      cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost * a_change_v_ego, jerk_comf * jerk_factor * J_EGO_COST * j_ego_v_ego]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, danger_cost]
    elif self.mode == 'blended':
      a_change_cost = 40.0 if prev_accel_constraint else 0
      #cost_weights = [0., 0.1, 0.2, 5.0, a_change_cost * a_change_v_ego, 1.0]
      # ✅ 如果是 e2e 主導，增加 MPC 對軌跡貼合懲罰（例如貼近模型預測軌跡）
      if self.source == 'e2e':
        x_weight = 2.5#1.5  # 原本可能是 0.1，加強貼合程度
        x_obstacle_weight = 0.5
        jerk_gain = 0.1
        a_change_gain = 0.1
      else:
        x_weight = 1.5#0.1
        x_obstacle_weight = 0.5#0.0
        jerk_gain = 1.0
        a_change_gain = 1.0
        
      if v_ego <= mid_thr:
        j_ego_v_ego *= 1.0  # 強化低速舒適性 20
      #cost_weights = [0., 0.1, 0.2, 5.0, a_change_cost * a_change_v_ego, 2.5 * j_ego_v_ego]
      cost_weights = [x_obstacle_weight, x_weight, 0.2, 5.0, a_change_cost * a_change_v_ego * a_change_gain, 1.0 * j_ego_v_ego * jerk_gain]
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
  #def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, v_ego):
    # 限制 a_lead_tau 穩定範圍，避免數值爆炸
    #a_lead_tau = np.clip(a_lead_tau, 1e-2, 10.0)
    a_lead_tau = np.clip(a_lead_tau, 0.1, 4.0)
    #================================================================
    # 停止狀態下，高靈敏預測（如 Stop & Go）
    if v_ego <= mid_thr:
      # 若前車真的明顯在啟動，允許快速起步
      #if v_lead < 1.0:
      if a_lead > 0.2:
        sensitivity_gain = 4.0 # 起步靈敏
      else:
        sensitivity_gain = 3.0 # 煞車靈敏
      a_lead_traj = a_lead * np.exp(-sensitivity_gain * a_lead_tau * (T_IDXS**2) / 2.)
    # 壅塞狀態（低速密集跟車）
    elif v_ego <= high_thr:
      sensitivity_gain = 2.0
      a_lead_traj = a_lead * np.exp(-sensitivity_gain * a_lead_tau * (T_IDXS**2) / 2.)

    # 一般高速巡航
    else:
      a_lead_traj = a_lead * np.exp(-T_IDXS / a_lead_tau)
    #======================要碼掉，同時要將line446 -> 445
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = np.nan_to_num(lead.vLead, nan=0.0)
      a_lead = np.nan_to_num(lead.aLeadK, nan=0.0)
      a_lead_tau = np.nan_to_num(lead.aLeadTau, nan=_LEAD_ACCEL_TAU)
      #a_lead_tau = np.clip(a_lead_tau, 1e-2, 10.0)
      a_lead_tau = np.clip(a_lead_tau, 0.1, 4.0)  
    else:
      # Fake a fast lead car, so MPC can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU
    
    min_brake = -ACCEL_MIN * 2  # 正数
    min_x = max(((v_ego + v_lead) / 2) * (v_ego - v_lead) / max(min_brake, 1e-3), 0.0)
    x_lead = np.clip(x_lead, min_x, np.inf)
    
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)

    #lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, v_ego)
    return lead_xv

  def update(self, radarstate, v_cruise, x, v, a, j, personality=log.LongitudinalPersonality.standard, dynamic_follow=False):
    v_ego = self.x0[1]
    a_lead0 = np.nan_to_num(radarstate.leadOne.aLeadK, nan=0.0) if radarstate.leadOne.status else 0.0
    a_lead1 = np.nan_to_num(radarstate.leadTwo.aLeadK, nan=0.0) if radarstate.leadTwo.status else 0.0
    a_lead_min = min(a_lead0, a_lead1)
    t_follow = get_adaptive_T_FOLLOW(v_ego, a_lead_min, personality)
    stop_distance = get_STOP_DISTANCE(personality)

    if self.params_store.get_bool("ToyotaTune") and not (self.CP.flags & ToyotaFlags.SMART_DSU):
      stop_distance += 3.0

    self.status = radarstate.leadOne.status or radarstate.leadTwo.status

    lead_xv_0 = self.process_lead(radarstate.leadOne)
    lead_xv_1 = self.process_lead(radarstate.leadTwo)
    
    # 模型4 原始碼（整合 lead obstacle 最小安全距離保護
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1], v_ego)
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1], v_ego)

    self.params[:,0] = ACCEL_MIN
    self.params[:,1] = ACCEL_MAX

    #===================================================================
    # 讀當前速度
    v_ego = self.x0[1]
    # 上一週期速度
    v_prev = self.prev_v_ego
    dv = v_ego - v_prev
    stopped_thr = 0.5          # 視為「靜止」的速度阈值

    #==================================================================
    #if self.mode == 'blended' and ((dv < 0 and v_ego <= low_thr) or v_ego > high_thr):
        #self.mode = 'acc'
        #self.set_weights(prev_accel_constraint=True, personality=personality, v_lead0=a_lead0, v_lead1=a_lead1)
    ##===elif self.mode == 'acc' and self.prev_v_ego <= stopped_thr and dv > 0:
    ##===elif self.mode == 'acc' and dv > 0 and (self.prev_v_ego <= stopped_thr or (stopped_thr < v_ego < high_thr)):
    #elif self.mode == 'acc' and ((self.prev_v_ego <= stopped_thr and dv > 0) or (stopped_thr < v_ego < high_thr and (a_lead0 > 0.3 or a_lead1 > 0.3))):
        #self.mode = 'blended'
        #self.set_weights(prev_accel_constraint=True, personality=personality, v_lead0=a_lead0, v_lead1=a_lead1)
    #==================================================================
    if v_ego > mid_thr:
        self.mode = 'acc'
        self.set_weights(prev_accel_constraint=True, personality=personality, v_lead0=a_lead0, v_lead1=a_lead1)
    elif v_ego <= mid_thr:
        self.mode = 'blended'
        self.set_weights(prev_accel_constraint=True, personality=personality, v_lead0=a_lead0, v_lead1=a_lead1)

    #==================================================================
    # 更新 prev_v_ego，供下一次使用
    self.prev_v_ego = v_ego
    #===================================================================

    if self.mode == 'acc':
      #self.params[:,5] = 0.85
      danger_factor = get_lead_danger_factor(v_ego)
      self.params[:,5] = danger_factor
      v_lower = v_ego + (T_IDXS * CRUISE_MIN_ACCEL * 0.95)
      v_upper = v_ego + (T_IDXS * CRUISE_MAX_ACCEL * 0.9)
      v_cruise_clipped = np.clip(v_cruise * np.ones(N+1), v_lower, v_upper)
      cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow, stop_distance)
      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
      self.source = SOURCES[np.argmin(x_obstacles[0])]

      x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

    elif self.mode == 'blended':
      danger_factor = get_lead_danger_factor(v_ego)
      self.params[:,5] = danger_factor
      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle])
      
      # cruise 目標距離（略為積極）
      cruise_target = T_IDXS * np.clip(v_cruise, v_ego - 4.0, 1e3) + x[0]
      
      # e2e 預測距離
      xforward = ((v[1:] + v[:-1]) / 2) * (T_IDXS[1:] - T_IDXS[:-1])
      x_e2e = np.cumsum(np.insert(xforward, 0, x[0]))
      #平滑濾波
      self.x_e2e_smooth = 0.8 * self.x_e2e_smooth + 0.2 * x_e2e if hasattr(self, "x_e2e_smooth") else x_e2e.copy()
      x_e2e = self.x_e2e_smooth
      # 混合 e2e 和 cruise，根據速度平滑插值
      v_low, v_high = 0.5, mid_thr
      w = np.clip((v_ego - v_low) / (v_high - v_low), 0.0, 0.4)
      #x_mixed = (1 - w) * np.minimum(x_e2e, cruise_target) + w * np.maximum(x_e2e, cruise_target)
      #x_mixed = 0.3 * np.minimum(x_e2e, cruise_target) + 0.7 * np.maximum(x_e2e, cruise_target)
      x_mixed = 0.2 * np.minimum(x_e2e, cruise_target) + 0.8 * np.maximum(x_e2e, cruise_target)
      #x_mixed = w * np.minimum(x_e2e, cruise_target) + (1 - w) * np.maximum(x_e2e, cruise_target)
      
      #x[:] = x_mixed  # 修正此行
      #提前計算 e2e 與 cruise 預測距離（source 決策用）
      e2e_dist = x_e2e[1]
      cruise_dist = cruise_target[1]

      #若低速且前車未加速 → 保留 e2e；否則用混合
      lead_accel = (a_lead0 > 0.2 and radarstate.leadOne.status) or \
                     (a_lead1 > 0.2 and radarstate.leadTwo.status)
      
      # ✅ 決定使用 e2e 或 x_mixed 軌跡
      if v_ego <= mid_thr and not lead_accel:
      #if v_ego <= mid_thr:
        x[:] = x_e2e
        self.source = 'e2e'
      else:
        x[:] = x_mixed
      
      # 更新 MPC yref
      self.yref[:,1] = x
      self.yref[:,2] = v
      self.yref[:,3] = a
      self.yref[:,5] = j
      for i in range(N):
        self.solver.set(i, "yref", self.yref[i])
      self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner update')

    if self.mode != 'blended':
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

    if self.mode == 'blended':
      if any((lead_0_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow, stop_distance)) - self.x_sol[:,0] < 0.0):
        self.source = 'lead0'
      if any((lead_1_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], t_follow, stop_distance)) - self.x_sol[:,0] < 0.0) and \
         (lead_1_obstacle[0] - lead_0_obstacle[0]) < 0:
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
