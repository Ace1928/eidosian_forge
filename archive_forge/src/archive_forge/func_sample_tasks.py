from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
def sample_tasks(self, n_tasks):
    return np.random.uniform(low=0.5, high=2.0, size=(n_tasks,))