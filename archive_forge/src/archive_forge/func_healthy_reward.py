import numpy as np
from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box
@property
def healthy_reward(self):
    return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward