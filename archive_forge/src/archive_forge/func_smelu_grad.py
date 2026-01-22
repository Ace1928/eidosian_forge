import math
from typing import Optional
import triton
import triton.language as tl
from xformers.components import Activation
@triton.jit
def smelu_grad(x):
    beta = 2.0
    relu_grad = tl.where(x >= beta, 1.0, 0.0)
    return tl.where(tl.abs(x) <= beta, (beta + x) / (2.0 * beta), relu_grad)