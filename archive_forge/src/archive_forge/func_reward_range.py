import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
@property
def reward_range(self):
    spec = self._env.reward_spec()
    if isinstance(spec, specs.BoundedArray):
        return (spec.minimum, spec.maximum)
    return (-float('inf'), float('inf'))