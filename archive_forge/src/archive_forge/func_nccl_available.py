import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def nccl_available():
    return _NCCL_AVAILABLE