from typing import Any, List, Optional, Tuple, Union
import numpy as np
import gym
from gym.vector.utils.spaces import batch_space
def call_wait(self, **kwargs) -> List[Any]:
    """After calling a method in :meth:`call_async`, this function collects the results."""