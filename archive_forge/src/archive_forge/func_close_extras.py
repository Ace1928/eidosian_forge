from typing import Any, List, Optional, Tuple, Union
import numpy as np
import gym
from gym.vector.utils.spaces import batch_space
def close_extras(self, **kwargs):
    return self.env.close_extras(**kwargs)