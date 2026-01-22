import gymnasium as gym
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
def from_jsonable(self, sample_n):
    return [np.asarray(sample) for sample in sample_n]