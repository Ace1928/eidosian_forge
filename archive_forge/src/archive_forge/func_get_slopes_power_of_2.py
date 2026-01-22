import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def get_slopes_power_of_2(n: int) -> List[float]:
    start = 2 ** (-2 ** (-(math.log2(n) - 3)))
    ratio = start
    return [start * ratio ** i for i in range(n)]