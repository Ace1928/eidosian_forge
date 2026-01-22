import random
import string
import gym
import gym.spaces
import numpy as np
import random
from collections import OrderedDict
from typing import List
import gym
import logging
import gym.spaces
import numpy as np
import collections
import warnings
import abc
def unmap_mixed(self, x: np.ndarray, aux: OrderedDict):
    unmapped = collections.OrderedDict()
    cur_index = 0
    for k, v in self.spaces.items():
        if v.is_flattenable():
            try:
                unmapped[k] = v.unmap_mixed(x[..., cur_index:cur_index + v.flattened.shape[0]], aux[k])
            except (KeyError, AttributeError):
                unmapped[k] = v.unmap(x[..., cur_index:cur_index + v.flattened.shape[0]])
            cur_index += v.flattened.shape[0]
        else:
            unmapped[k] = aux[k]
    return unmapped