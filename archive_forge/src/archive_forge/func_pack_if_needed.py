from ray.rllib.utils.annotations import DeveloperAPI
import logging
import time
import base64
import numpy as np
from ray import cloudpickle as pickle
@DeveloperAPI
def pack_if_needed(data):
    if isinstance(data, np.ndarray):
        data = pack(data)
    return data