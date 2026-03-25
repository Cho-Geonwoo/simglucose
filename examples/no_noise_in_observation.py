import gym

from gym.envs.registration import register

register(
    id="simglucose-adolescent2-v0",
    entry_point="simglucose.envs:T1DSimEnv",
    kwargs={
        "patient_name": "adolescent#002",
        "use_noise": False,
    },
)

env = gym.make("simglucose-adolescent2-v0")
