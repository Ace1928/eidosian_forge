import math
from enum import Enum
from typing import Optional
import triton
import triton.language as tl
def get_triton_activation_kernel(activation: Optional[Activation]):
    return {Activation.ReLU: relu, Activation.LeakyReLU: leaky_relu, Activation.GeLU: gelu, Activation.GeLUApprox: gelu_approx, Activation.SquaredReLU: squared_relu}[activation] if activation else None