from typing import Optional
import torch
import triton
import triton.language as tl
from xformers.triton.k_activations import (

    Compute grad_in = activation^-1(grad_out) @ weight.transpose()

    .. note: The weight buffer is transposed on the fly
    .. note: Activation gradient needs to be a Triton kernel
    