import datetime as dt
import unittest

import gym

from simglucose.simulation.scenario import CustomScenario


class TestAutoBolus(unittest.TestCase):
    def test_auto_bolus_follows_split_meal_consumption(self):
        from gym.envs.registration import register

        env_id = "simglucose-auto-bolus-v0"
        start_time = dt.datetime(2018, 1, 1, 0, 0, 0)
        custom_meal_scenario = CustomScenario(start_time=start_time, scenario=[(1, 30)])

        register(
            id=env_id,
            entry_point="simglucose.envs:T1DSimEnv",
            kwargs={
                "patient_name": "adult#001",
                "custom_scenario": custom_meal_scenario,
                "sample_time": 1.0,
                "interaction_step": 1.0,
                "auto_bolus": True,
                "carbohydrate_ratio": 10.0,
            },
        )

        env = gym.make(env_id)
        env.reset()

        auto_bolus_series = []
        for _ in range(80):
            _, _, done, info = env.step(0.0)
            auto_bolus_series.append(float(info.get("auto_bolus_insulin", 0.0)))
            if done:
                break

        # One 30g meal with EAT_RATE=5 g/min and CR=10 should generate
        # multiple steps of positive split bolus.
        positive_steps = [x for x in auto_bolus_series if x > 0]

        self.assertTrue(len(positive_steps) >= 5)
        self.assertTrue(any(x >= 0.4 for x in positive_steps))


if __name__ == "__main__":
    unittest.main()
