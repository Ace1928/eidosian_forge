import math
from typing import Optional
import triton
import triton.language as tl
from xformers.components import Activation
@triton.jit
def star_relu_grad(x):
    return tl.where(x >= 0.0, 1.7888 * x, 0.0)