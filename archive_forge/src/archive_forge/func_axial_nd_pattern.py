import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def axial_nd_pattern(*sizes):
    d = local_nd_distance(*sizes, p=0)
    return d < 2