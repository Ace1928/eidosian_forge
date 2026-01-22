import math
from enum import Enum
from typing import Optional
import triton
import triton.language as tl

    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    