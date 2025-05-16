import time
import math
from collections import deque

# Constants for lateral acceleration and minimum speed
target_lat_a = 2.3  # m/s^2
min_target_v = 5.0  # m/s

class VisionTurnController:
    def __init__(self,
                 history_len=10,
                 nominal_fps=20.0,
                 filter_alpha=0.5,
                 smoothing_alpha=0.3):
        # Parameters
        self.history_len = history_len
        self.nominal_fps = nominal_fps
        self.filter_alpha = filter_alpha
        self.smoothing_alpha = smoothing_alpha

        # State
        self.curvature_history = deque(maxlen=self.history_len)
        self.filtered_curvatures = {}
        self.last_time = None
        self.last_frame = None
        self.v_target = min_target_v
        self.max_pred_lat_acc = target_lat_a

    def update(self, curvatures: dict, frame: int, v_ego: float, curr_time: float = None) -> float:
        """
        Update the target speed based on visual curvature estimates.

        Args:
            curvatures: dict mapping object IDs to measured curvature (1/m).
            frame: current frame index
            v_ego: current vehicle speed (m/s)
            curr_time: timestamp of this update (seconds)

        Returns:
            Smoothed target speed (m/s)
        """
        # Use system time if not provided
        if curr_time is None:
            curr_time = time.time()

        # Skip if same frame
        if frame == self.last_frame:
            return self.v_target

        # Compute dt, with fallback to nominal FPS
        if self.last_time is None:
            dt = 1.0 / self.nominal_fps
        else:
            dt = curr_time - self.last_time
            if dt <= 0:
                dt = 1.0 / self.nominal_fps

        # 1. Noise filtering: exponential moving average per ID
        for obj_id, meas in curvatures.items():
            prev = self.filtered_curvatures.get(obj_id, meas)
            filtered = self.filter_alpha * meas + (1 - self.filter_alpha) * prev
            self.filtered_curvatures[obj_id] = filtered

        # Compute average filtered curvature
        if self.filtered_curvatures:
            avg_curvature = sum(self.filtered_curvatures.values()) / len(self.filtered_curvatures)
        else:
            # No data: fallback to last known curvature or zero
            avg_curvature = self.curvature_history[-1] if self.curvature_history else 0.0

        # 2. Record history (with automatic length limit)
        self.curvature_history.append(avg_curvature)

        # 3. Linear extrapolation for curvature rate
        if len(self.curvature_history) >= 2:
            prev_curv = self.curvature_history[-2]
            predicted_rate = (avg_curvature - prev_curv) / dt
        else:
            predicted_rate = 0.0

        # 4. Dynamic lateral acceleration limit based on predicted rate
        self.max_pred_lat_acc = target_lat_a / (1 + abs(predicted_rate))

        # 5. Compute raw target speed based on scaling current speed
        #    v_target_raw = v_ego * sqrt(target_lat_a / max_pred_lat_acc)
        v_target_raw = v_ego * math.sqrt(target_lat_a / self.max_pred_lat_acc)
        v_target_raw = max(v_target_raw, min_target_v)

        # 6. Smooth target speed changes
        self.v_target = (self.smoothing_alpha * v_target_raw
                         + (1 - self.smoothing_alpha) * self.v_target)

        # Update time/frame and return
        self.last_time = curr_time
        self.last_frame = frame
        return self.v_target
