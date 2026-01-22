from typing import Optional
import torch
import triton
import triton.language as tl
from xformers.triton.k_activations import (

    Compute e = activation(x @ weight + bias).
    This wrapper kicks the `kernel_fma` Triton kernel
    