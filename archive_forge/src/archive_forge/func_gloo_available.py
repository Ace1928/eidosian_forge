import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def gloo_available():
    return _GLOO_AVAILABLE