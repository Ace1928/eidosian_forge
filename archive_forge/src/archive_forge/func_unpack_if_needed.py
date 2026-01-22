from ray.rllib.utils.annotations import DeveloperAPI
import logging
import time
import base64
import numpy as np
from ray import cloudpickle as pickle
@DeveloperAPI
def unpack_if_needed(data):
    if is_compressed(data):
        data = unpack(data)
    return data