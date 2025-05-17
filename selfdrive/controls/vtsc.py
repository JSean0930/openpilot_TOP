import time
import math
from collections import deque

# 橫向加速度與最低速度的常數
target_lat_a = 2.3  # 公尺/平方秒
min_target_v = 5.0  # 公尺/秒

class VisionTurnController:
    def __init__(self,
                 history_len=10,
                 nominal_fps=20.0,
                 filter_alpha=0.5,
                 smoothing_alpha=0.3):
        # 參數
        self.history_len = history_len
        self.nominal_fps = nominal_fps
        self.filter_alpha = filter_alpha
        self.smoothing_alpha = smoothing_alpha

        # 狀態
        self.curvature_history = deque(maxlen=self.history_len)
        self.filtered_curvatures = {}
        self.last_time = None
        self.last_frame = None
        self.v_target = min_target_v
        self.max_pred_lat_acc = target_lat_a

    def update(self, curvatures: dict, frame: int, v_ego: float, curr_time: float = None) -> float:
        """
        根據視覺曲率估計更新目標速度。

        參數:
            curvatures: dict，將物件 ID 映射到測量到的曲率 (1/公尺)。
            frame: 當前幀索引
            v_ego: 當前車速 (公尺/秒)
            curr_time: 此次更新的時間戳 (秒)

        回傳:
            平滑後的目標速度 (公尺/秒)
        """
        # 如果未提供，使用系統時間
        if curr_time is None:
            curr_time = time.time()

        # 如果幀未改變，跳過
        if frame == self.last_frame:
            return self.v_target

        # 計算 dt，若無歷史時間則使用預設幀率
        if self.last_time is None:
            dt = 1.0 / self.nominal_fps
        else:
            dt = curr_time - self.last_time
            if dt <= 0:
                dt = 1.0 / self.nominal_fps

        # 1. 雜訊過濾：對每個 ID 使用指數移動平均
        for obj_id, meas in curvatures.items():
            prev = self.filtered_curvatures.get(obj_id, meas)
            filtered = self.filter_alpha * meas + (1 - self.filter_alpha) * prev
            self.filtered_curvatures[obj_id] = filtered

        # 計算平均過濾後的曲率
        if self.filtered_curvatures:
            avg_curvature = sum(self.filtered_curvatures.values()) / len(self.filtered_curvatures)
        else:
            # 無數據：回退到最後已知的曲率或 0
            avg_curvature = self.curvature_history[-1] if self.curvature_history else 0.0

        # 2. 記錄歷史曲率 (自動限制長度)
        self.curvature_history.append(avg_curvature)

        # 3. 線性外插計算曲率變化率
        if len(self.curvature_history) >= 2:
            prev_curv = self.curvature_history[-2]
            predicted_rate = (avg_curvature - prev_curv) / dt
        else:
            predicted_rate = 0.0

        # 4. 根據預測變化率動態調整橫向加速度上限
        self.max_pred_lat_acc = target_lat_a / (1 + abs(predicted_rate))

        # 5. 根據當前狀態計算原始目標速度
        #    v_target_raw = v_ego * sqrt(target_lat_a / max_pred_lat_acc)
        v_target_raw = v_ego * math.sqrt(target_lat_a / self.max_pred_lat_acc)
        v_target_raw = max(v_target_raw, min_target_v)

        # 6. 平滑目標速度變化
        self.v_target = (self.smoothing_alpha * v_target_raw
                         + (1 - self.smoothing_alpha) * self.v_target)

        # 更新時間/幀並回傳
        self.last_time = curr_time
        self.last_frame = frame

        return self.v_target
