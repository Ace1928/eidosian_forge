import math
from enum import Enum
from typing import Optional
import triton
import triton.language as tl
@triton.jit
def squared_relu_grad(x):
    return tl.where(x >= 0, 2.0 * x, 0.0)