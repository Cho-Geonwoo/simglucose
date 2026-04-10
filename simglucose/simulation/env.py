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
    def __init__(
        self,
        patient,
        sensor,
        pump,
        scenario,
        interaction_step=3.0,
        auto_bolus=False,
        carbohydrate_ratio=None,
    ):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self.interaction_step = interaction_step
        self.auto_bolus = bool(auto_bolus)
        self.carbohydrate_ratio = carbohydrate_ratio

        if self.auto_bolus:
            if self.carbohydrate_ratio is None:
                raise ValueError(
                    "carbohydrate_ratio is required when auto_bolus is enabled"
                )
            if self.carbohydrate_ratio <= 0:
                raise ValueError("carbohydrate_ratio must be positive")

        self._reset()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def _predict_consumed_cho(self, announced_cho):
        pending_meal = self.patient.planned_meal + announced_cho
        if pending_meal <= 0:
            return 0.0

        return float(
            min(self.patient.EAT_RATE * self.patient.sample_time, pending_meal)
        )

    def mini_step(self, action, update_observation=False):
        # current action
        patient_action = self.scenario.get_action(self.time)
        announced_cho = float(patient_action.meal)
        consumed_cho = self._predict_consumed_cho(announced_cho)

        auto_bolus_input = 0.0
        if self.auto_bolus:
            # One-shot bolus at meal announce: add full meal/CR to pending pool.
            if announced_cho > 0:
                self._pending_bolus_units += announced_cho / self.carbohydrate_ratio

            if self._pending_bolus_units > 0:
                # Deliver as much as pump allows this step; carry remainder forward.
                max_deliverable = (
                    self.pump._params["max_bolus"] * self.patient.sample_time
                )
                deliver_units = min(self._pending_bolus_units, max_deliverable)
                self._pending_bolus_units -= deliver_units
                self._pending_bolus_units = max(0.0, self._pending_bolus_units)
                # Convert from U (total this step) to U/min rate for simulator.
                auto_bolus_input = deliver_units / self.patient.sample_time

        manual_bolus_input = 0.0 if self.auto_bolus else float(action.bolus)
        final_bolus_input = manual_bolus_input + auto_bolus_input

        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(final_bolus_input)
        auto_bolus = self.pump.bolus(auto_bolus_input) if self.auto_bolus else 0.0
        insulin = basal + bolus
        CHO = consumed_cho
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient, update_observation=update_observation)

        return CHO, insulin, BG, CGM, basal, bolus, auto_bolus, announced_cho

    def step(self, action, reward_fun=risk_diff):
        """
        action is a namedtuple with keys: basal, bolus
        """
        total_cho = 0.0
        average_insulin = 0.0
        average_basal = 0.0
        average_bolus = 0.0
        average_auto_bolus = 0.0
        average_bg = 0.0
        average_cgm = 0.0
        total_announced_cho = 0.0
        cho_list = []
        announced_cho_list = []
        insulin_list = []
        basal_insulin_list = []
        bolus_insulin_list = []
        auto_bolus_insulin_list = []
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
            (
                tmp_CHO,
                tmp_insulin,
                tmp_BG,
                tmp_CGM,
                tmp_basal,
                tmp_bolus,
                tmp_auto_bolus,
                tmp_announced_cho,
            ) = self.mini_step(action, update_observation)
            total_cho += tmp_CHO
            total_announced_cho += tmp_announced_cho
            average_insulin += tmp_insulin / self.interaction_step
            average_basal += tmp_basal / self.interaction_step
            average_bolus += tmp_bolus / self.interaction_step
            average_auto_bolus += tmp_auto_bolus / self.interaction_step
            average_bg += tmp_BG / self.interaction_step
            average_cgm += tmp_CGM / self.interaction_step
            cho_list.append(tmp_CHO)
            announced_cho_list.append(tmp_announced_cho)
            insulin_list.append(tmp_insulin)
            basal_insulin_list.append(tmp_basal)
            bolus_insulin_list.append(tmp_bolus)
            auto_bolus_insulin_list.append(tmp_auto_bolus)
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
        self.CHO_hist.append(total_cho)
        self.insulin_hist.append(average_insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(average_bg)
        self.CGM_hist.append(average_cgm)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        BG_last_hour = self.CGM_hist[-2:]
        reward = reward_fun(BG_last_hour)
        done = any(bg < 10 or bg > 600 for bg in bg_list)
        obs = Observation(CGM=self.CGM_hist[-1])

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=total_cho,
            announced_meal=total_announced_cho,
            patient_state=self.patient.state,
            time=self.time,
            bg=average_bg,
            lbgi=LBGI,
            hbgi=HBGI,
            risk=risk,
            cho_list=cho_list,
            announced_cho_list=announced_cho_list,
            insulin_list=insulin_list,
            basal_insulin=average_basal,
            bolus_insulin=average_bolus,
            auto_bolus_insulin=average_auto_bolus,
            basal_insulin_list=basal_insulin_list,
            bolus_insulin_list=bolus_insulin_list,
            auto_bolus_insulin_list=auto_bolus_insulin_list,
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
        self._pending_bolus_units = 0.0

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
            announced_meal=0,
            patient_state=self.patient.state,
            time=self.time,
            bg=self.BG_hist[0],
            lbgi=self.LBGI_hist[0],
            hbgi=self.HBGI_hist[0],
            risk=self.risk_hist[0],
            cho_list=[],
            announced_cho_list=[],
            insulin_list=[],
            basal_insulin=0.0,
            bolus_insulin=0.0,
            auto_bolus_insulin=0.0,
            basal_insulin_list=[],
            bolus_insulin_list=[],
            auto_bolus_insulin_list=[],
            bg_list=[],
            cgm_list=[],
            lbgi_list=[],
            hbgi_list=[],
            risk_list=[],
            datetime_list=[],
        )

    def render(self, close=False):
        if close:
            self._close_viewer()
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def _close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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
