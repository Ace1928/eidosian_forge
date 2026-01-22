import math
from typing import Optional
import triton
import triton.language as tl
from xformers.components import Activation
@triton.jit
def smelu(x):
    """
    SmeLU_ activation -  Smooth ReLU with beta=2.0

    .. _SmeLU: https://arxiv.org/pdf/2202.06499.pdf
    """
    beta = 2.0
    relu = tl.where(x >= beta, x, 0.0)
    return tl.where(tl.abs(x) <= beta, (x + beta) * (x + beta) / (4.0 * beta), relu)