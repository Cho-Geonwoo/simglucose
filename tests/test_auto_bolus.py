import datetime as dt
import sys
import types
import unittest
from collections import namedtuple

if "pkg_resources" not in sys.modules:
    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.resource_filename = lambda *args, **kwargs: ""
    sys.modules["pkg_resources"] = pkg_resources

from simglucose.simulation.env import T1DSimEnv


Observation = namedtuple("Observation", ["Gsub"])
ScenarioAction = namedtuple("ScenarioAction", ["meal"])
ControllerAction = namedtuple("Action", ["basal", "bolus"])
ZERO_ACTION = ControllerAction(0.0, 0.0)


class DummyPatient:
    def __init__(self, sample_time, bg=180.0):
        self.SAMPLE_TIME = sample_time
        self.observation = Observation(Gsub=bg)
        self.t = 0.0
        self.state = None
        self.name = "dummy"

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action):
        self.t += self.sample_time

    def reset(self):
        self.t = 0.0


class DummySensor:
    def measure(self, patient, update_observation=False):
        return patient.observation.Gsub

    def reset(self):
        pass


class DummyPump:
    def __init__(self):
        self._params = {"max_bolus": 30.0}

    def basal(self, amount):
        return float(amount)

    def bolus(self, amount):
        amount = max(0.0, min(float(amount), self._params["max_bolus"]))
        return round(amount / 0.05) * 0.05

    def reset(self):
        pass


class DummyScenario:
    def __init__(self, start_time, meal_schedule):
        self.start_time = start_time
        self.meal_schedule = meal_schedule

    def get_action(self, t):
        return ScenarioAction(meal=self.meal_schedule.get(t, 0.0))

    def reset(self):
        pass


def build_env(sample_time, interaction_step, meal_schedule):
    start_time = dt.datetime(2018, 1, 1, 0, 0, 0)
    return T1DSimEnv(
        patient=DummyPatient(sample_time=sample_time),
        sensor=DummySensor(),
        pump=DummyPump(),
        scenario=DummyScenario(start_time=start_time, meal_schedule=meal_schedule),
        interaction_step=interaction_step,
        auto_bolus=True,
        carbohydrate_ratio=10.0,
    )


def build_env_with_params(
    sample_time,
    interaction_step,
    meal_schedule,
    carbohydrate_ratio=10.0,
    correction_factor=None,
    target_blood_glucose=144.0,
    bg=180.0,
):
    start_time = dt.datetime(2018, 1, 1, 0, 0, 0)
    return T1DSimEnv(
        patient=DummyPatient(sample_time=sample_time, bg=bg),
        sensor=DummySensor(),
        pump=DummyPump(),
        scenario=DummyScenario(start_time=start_time, meal_schedule=meal_schedule),
        interaction_step=interaction_step,
        auto_bolus=True,
        carbohydrate_ratio=carbohydrate_ratio,
        correction_factor=correction_factor,
        target_blood_glucose=target_blood_glucose,
    )


def run_env_steps(env, steps):
    infos = []
    for _ in range(steps):
        _, _, _, info = env.step(ZERO_ACTION)
        infos.append(info)
    return infos


def total_auto_bolus_units(infos, sample_time):
    return sum(
        float(rate) * float(sample_time)
        for info in infos
        for rate in info["auto_bolus_insulin_list"]
    )


class TestAutoBolus(unittest.TestCase):
    def test_auto_bolus_applies_on_next_interaction_step(self):
        start_time = dt.datetime(2018, 1, 1, 0, 0, 0)
        env = build_env(
            sample_time=1.0,
            interaction_step=3.0,
            meal_schedule={start_time: 30.0},
        )

        env.reset()
        info1, info2, info3 = run_env_steps(env, 3)

        self.assertEqual(info1["meal"], 30.0)
        self.assertEqual(info1["auto_bolus_insulin"], 0.0)
        self.assertEqual(info2["auto_bolus_insulin"], 1.0)
        self.assertEqual(info2["auto_bolus_insulin_list"], [1.0, 1.0, 1.0])
        self.assertEqual(info3["auto_bolus_insulin"], 0.0)

    def test_auto_bolus_preserves_total_units_across_interaction_cadences(self):
        start_time = dt.datetime(2018, 1, 1, 0, 0, 0)
        test_cases = [
            {
                "name": "3min interaction",
                "sample_time": 1.0,
                "interaction_step": 3.0,
                "steps": 3,
                "expected_active_ministeps": 3,
            },
            {
                "name": "1min interaction",
                "sample_time": 1.0,
                "interaction_step": 1.0,
                "steps": 5,
                "expected_active_ministeps": 3,
            },
            {
                "name": "10s interaction",
                "sample_time": 10.0 / 60.0,
                "interaction_step": 1.0,
                "steps": 20,
                "expected_active_ministeps": 18,
            },
            {
                "name": "1s interaction",
                "sample_time": 1.0 / 60.0,
                "interaction_step": 1.0,
                "steps": 182,
                "expected_active_ministeps": 180,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                env = build_env(
                    sample_time=case["sample_time"],
                    interaction_step=case["interaction_step"],
                    meal_schedule={start_time: 30.0},
                )
                env.reset()
                infos = run_env_steps(env, case["steps"])

                active_rates = [
                    float(rate)
                    for info in infos
                    for rate in info["auto_bolus_insulin_list"]
                    if float(rate) > 0
                ]

                self.assertEqual(infos[0]["meal"], 30.0)
                self.assertEqual(infos[0]["auto_bolus_insulin"], 0.0)
                self.assertEqual(
                    len(active_rates),
                    case["expected_active_ministeps"],
                )
                self.assertTrue(all(abs(rate - 1.0) < 1e-6 for rate in active_rates))
                self.assertAlmostEqual(
                    total_auto_bolus_units(infos, case["sample_time"]),
                    3.0,
                    places=6,
                )

    def test_correction_factor_applies_when_recent_meal_history_is_empty(self):
        start_time = dt.datetime(2018, 1, 1, 0, 0, 0)
        env = build_env_with_params(
            sample_time=1.0,
            interaction_step=1.0,
            meal_schedule={start_time: 30.0},
            carbohydrate_ratio=10.0,
            correction_factor=30.0,
            target_blood_glucose=144.0,
            bg=180.0,
        )

        env.reset()
        info1, info2, info3, info4, info5 = run_env_steps(env, 5)

        self.assertEqual(info1["meal"], 30.0)
        self.assertEqual(info1["auto_bolus_insulin"], 0.0)
        self.assertAlmostEqual(info2["auto_bolus_insulin"], 1.4)
        self.assertAlmostEqual(info3["auto_bolus_insulin"], 1.4)
        self.assertAlmostEqual(info4["auto_bolus_insulin"], 1.4)
        self.assertEqual(info5["auto_bolus_insulin"], 0.0)
        self.assertAlmostEqual(total_auto_bolus_units([info2, info3, info4], 1.0), 4.2)

    def test_correction_factor_is_skipped_when_recent_meal_exists(self):
        start_time = dt.datetime(2018, 1, 1, 0, 0, 0)
        env = build_env_with_params(
            sample_time=1.0,
            interaction_step=1.0,
            meal_schedule={
                start_time: 30.0,
                start_time + dt.timedelta(minutes=10): 30.0,
            },
            carbohydrate_ratio=10.0,
            correction_factor=30.0,
            target_blood_glucose=144.0,
            bg=180.0,
        )

        env.reset()
        infos = run_env_steps(env, 15)

        second_window_rates = [
            float(info["auto_bolus_insulin"])
            for info in infos[11:14]
        ]
        self.assertEqual(second_window_rates, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
