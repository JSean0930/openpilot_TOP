#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from typing import Any

import capnp
from cereal import messaging, log, car
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame



class KalmanParams:
    def __init__(self, dt: float):
        assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.C = np.array([1.0, 0.0])
        dts = np.linspace(0.01, 0.2, 20)
        K0 = np.array([0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689, 
                       0.21372394, 0.22761098, 0.24069424, 0.253096, 0.26491023, 
                       0.27621103, 0.28705801, 0.29750003, 0.30757767, 0.31732515, 
                       0.32677158, 0.33594201, 0.34485814, 0.35353899, 0.36200124])
        K1 = np.array([0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 
                       0.28342219, 0.28144091, 0.27958406, 0.27783249, 0.27617149, 
                       0.27458948, 0.27307714, 0.27162685, 0.2702321, 0.26888707, 
                       0.26758677, 0.26632688, 0.26510363, 0.26391379, 0.26275457])
        self.K = np.vstack((K0, K1)).T


class Track:
    def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
        self.identifier = identifier
        self.cnt = 0
        self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
        self.K_A = kalman_params.A
        self.K_C = kalman_params.C
        self.K_K = kalman_params.K
        self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)
        self.prev_vLead = None

    def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float):
        # relative values, copy
        self.dRel = d_rel
        self.yRel = y_rel
        self.vRel = v_rel
        self.vLead = v_lead
        self.measured = measured

        # Update Kalman filter only if the speed significantly changes
        if self.prev_vLead is None or abs(self.vLead - self.prev_vLead) > 0.1:
            self.kf.update(self.vLead)
            self.prev_vLead = self.vLead

        self.vLeadK = float(self.kf.x[SPEED][0])
        self.aLeadK = float(self.kf.x[ACCEL][0])

        # Adjust acceleration decay time constant only when necessary
        if abs(self.aLeadK) < 0.5 and self.aLeadTau.x != _LEAD_ACCEL_TAU:
            self.aLeadTau.x = _LEAD_ACCEL_TAU

class RadarD:
  def __init__(self, delay: float = 0.0):
    self.current_time = 0.0

    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

    self.ready = False

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    # *** publish radarState ***
    self.radar_state_valid = sm.all_checks()
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      self.radar_state.leadOne = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego, low_speed_override=True)
      self.radar_state.leadTwo = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego, low_speed_override=False)

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)


# fuses camera and radar data for best lead detection
def main() -> None:
  config_realtime_process(5, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  # *** setup messaging
  sm = messaging.SubMaster(['modelV2', 'carState', 'liveTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState'])

  RD = RadarD(CP.radarDelay)

  while 1:
    sm.update()

    RD.update(sm, sm['liveTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
