# PFEIFER - VTSC Optimized

# Acknowledgements:
# Based on original VTSC implementation from move-fast and sunnypilot.

import numpy as np
from time import time
from collections import deque
from openpilot.common.params import Params

params = Params()

TARGET_LAT_A = 2.3  # m/s^2
MIN_TARGET_V = 5.0  # m/s
HISTORY_LENGTH = 10  # number of past curvature samples to average
SMOOTHING_ALPHA = 0.3  # for v_target exponential smoothing

class VisionTurnController:
    def __init__(self):
        self.op_enabled = False
        self.gas_pressed = False
        self.enabled = params.get_bool("TurnVisionControl")
        self.last_params_update = 0.0

        self.v_target = MIN_TARGET_V
        self.smoothed_v_target = MIN_TARGET_V

        # For curvature history and smoothing
        self.curvature_history = deque(maxlen=HISTORY_LENGTH)

        # For computing actual dt between updates
        self.last_time = time()

    @property
    def active(self):
        return self.op_enabled and not self.gas_pressed and self.enabled

    def update_params(self):
        t = time()
        if t > self.last_params_update + 5.0:
            self.enabled = params.get_bool("TurnVisionControl")
            self.last_params_update = t

    def update(self, op_enabled: bool, v_ego: float, sm: dict):
        # 1) Timing update for dt
        current_time = time()
        dt = current_time - self.last_time if self.last_time else 0.1
        self.last_time = current_time

        # 2) Update enable flags
        self.update_params()
        self.op_enabled = op_enabled
        self.gas_pressed = sm['carState'].gasPressed

        # 3) Extract model predictions
        rate_plan = np.abs(np.array(sm['modelV2'].orientationRate.z))
        vel_plan = np.array(sm['modelV2'].velocity.x)

        # 4) Validate prediction arrays
        if rate_plan.size == 0 or vel_plan.size == 0:
            return

        # 5) Compute curvature: yaw rate / velocity
        predicted_curvatures = rate_plan / np.maximum(vel_plan, 0.1)

        # 6) Filter extremes using percentile
        max_curve_sample = np.percentile(predicted_curvatures, 90)

        # 7) Maintain history buffer和預測未來曲率趨勢
        self.curvature_history.append(max_curve_sample)

        # 過去曲率值
        curve_array = np.array(self.curvature_history)
        if len(curve_array) >= 3:
            # 擬合一條曲率趨勢線 (線性回歸)
            x_hist = np.arange(len(curve_array))
            coeffs = np.polyfit(x_hist, curve_array, deg=1)
            future_curve = coeffs[0] * (len(curve_array) + 2) + coeffs[1]  # 預測兩步後的曲率
            avg_curve = max(np.mean(curve_array), future_curve)  # 選擇保守估計
        else:
            avg_curve = float(np.mean(curve_array))
    
        self.curvature_history.append(max_curve_sample)
        avg_curve = float(np.mean(self.curvature_history))

        # 8) Compute raw target velocity based on lateral accel limit
        v_ego_safe = max(v_ego, 0.1)
        if avg_curve > 0:
            v_target_raw = np.sqrt(TARGET_LAT_A / avg_curve)
        else:
            v_target_raw = MIN_TARGET_V
        v_target_raw = max(v_target_raw, MIN_TARGET_V)

        # 9) Smooth v_target to avoid abrupt changes
        self.smoothed_v_target = (
            SMOOTHING_ALPHA * self.smoothed_v_target
            + (1 - SMOOTHING_ALPHA) * v_target_raw
        )

        # 10) Final enforced target velocity
        self.v_target = max(self.smoothed_v_target, MIN_TARGET_V)

# Instantiate controller
vtsc = VisionTurnController()
