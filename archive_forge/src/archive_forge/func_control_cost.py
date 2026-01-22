import numpy as np
from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box
def control_cost(self, action):
    control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
    return control_cost