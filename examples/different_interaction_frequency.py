import gym

from gym.envs.registration import register

register(
    id="simglucose-child1-v0",
    entry_point="simglucose.envs:T1DSimEnv",
    kwargs={
        "patient_name": "child#001",
        "schedule": schedule,
        "sample_time": 1.0,
        "interaction_step": 3.0,
    },
)  # sample time: 1min, interaction step: 3min

register(
    id="simglucose-child1-v1",
    entry_point="simglucose.envs:T1DSimEnv",
    kwargs={
        "patient_name": "child#001",
        "schedule": schedule,
        "sample_time": 1.0 / 60.0,
        "interaction_step": 180,
    },
)  # sample time: 1s, interaction step: 3min

register(
    id="simglucose-child1-v2",
    entry_point="simglucose.envs:T1DSimEnv",
    kwargs={
        "patient_name": "child#001",
        "schedule": schedule,
        "sample_time": 10.0 / 60.0,
        "interaction_step": 18,
    },
)  # sample time: 10s, interaction step: 3min

env = gym.make("simglucose-child1-v0")
env = gym.make("simglucose-child1-v1")
env = gym.make("simglucose-child1-v2")
