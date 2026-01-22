import math
from typing import Optional
import triton
import triton.language as tl
from xformers.components import Activation
def get_triton_activation_index(activation: Optional[Activation]) -> int:
    return {Activation.ReLU: 1, Activation.LeakyReLU: 2, Activation.GeLU: 3, Activation.SquaredReLU: 4, Activation.SmeLU: 5, Activation.StarReLU: 6}[activation] if activation is not None else 0