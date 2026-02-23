from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


Observation = namedtuple("Observation", ["CGM"])
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


class T1DSimEnv(object):
    def __init__(self, patient, sensor, pump, scenario, interaction_step=3.0):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self.interaction_step = interaction_step
        self._reset()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action, update_observation=False):
        # current action
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient, update_observation=update_observation)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=risk_diff):
        """
        action is a namedtuple with keys: basal, bolus
        """
        average_cho = 0.0
        average_insulin = 0.0
        average_bg = 0.0
        average_cgm = 0.0
        cho_list = []
        insulin_list = []
        bg_list = []
        cgm_list = []
        lbgi_list = []
        hbgi_list = []
        risk_list = []
        datetime_list = []

        for i in range(int(self.interaction_step)):
            # TODO: change logic to use interaction_step
            update_observation = False
            if i == int(self.interaction_step) - 1:
                update_observation = True
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(
                action, update_observation
            )
            average_cho += tmp_CHO / self.interaction_step
            average_insulin += tmp_insulin / self.interaction_step
            average_bg += tmp_BG / self.interaction_step
            average_cgm += tmp_CGM / self.interaction_step
            cho_list.append(tmp_CHO)
            insulin_list.append(tmp_insulin)
            bg_list.append(tmp_BG)
            cgm_list.append(tmp_CGM)
            datetime_list.append(self.time)

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([average_bg], horizon)

        for i in range(int(self.interaction_step)):
            curr_lbgi, curr_hbgi, curr_risk = risk_index([bg_list[i]], horizon)
            lbgi_list.append(curr_lbgi)
            hbgi_list.append(curr_hbgi)
            risk_list.append(curr_risk)

        # Record current action
        self.CHO_hist.append(average_cho)
        self.insulin_hist.append(average_insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(average_bg)
        self.CGM_hist.append(average_cgm)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        BG_last_hour = self.CGM_hist[-2:]
        reward = reward_fun(BG_last_hour)

        # Harry: changed this to be similar to: https://arxiv.org/pdf/2010.06266.pdf
        # done = average_bg < 70 or average_bg > 350
        done = average_bg < 10 or average_bg > 1000
        obs = Observation(CGM=average_cgm)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=average_cho,
            patient_state=self.patient.state,
            time=self.time,
            bg=average_bg,
            lbgi=LBGI,
            hbgi=HBGI,
            risk=risk,
            cho_list=cho_list,
            insulin_list=insulin_list,
            bg_list=bg_list,
            cgm_list=cgm_list,
            lbgi_list=lbgi_list,
            hbgi_list=hbgi_list,
            risk_list=risk_list,
            datetime_list=datetime_list,
        )

    def _reset(self):
        self.sample_time = self.patient.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []

    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()

        self._reset()

        CGM = self.sensor.measure(self.patient)
        obs = Observation(CGM=CGM)
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state,
            time=self.time,
            bg=self.BG_hist[0],
            lbgi=self.LBGI_hist[0],
            hbgi=self.HBGI_hist[0],
            risk=self.risk_hist[0],
            cho_list=[],
            insulin_list=[],
            bg_list=[],
            cgm_list=[],
            lbgi_list=[],
            hbgi_list=[],
            risk_list=[],
            datetime_list=[],
        )

    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def show_history(self):
        df = pd.DataFrame()
        df["Time"] = pd.Series(self.time_hist)
        df["BG"] = pd.Series(self.BG_hist)
        df["CGM"] = pd.Series(self.CGM_hist)
        df["CHO"] = pd.Series(self.CHO_hist)
        df["insulin"] = pd.Series(self.insulin_hist)
        df["LBGI"] = pd.Series(self.LBGI_hist)
        df["HBGI"] = pd.Series(self.HBGI_hist)
        df["Risk"] = pd.Series(self.risk_hist)
        df = df.set_index("Time")
        return df
