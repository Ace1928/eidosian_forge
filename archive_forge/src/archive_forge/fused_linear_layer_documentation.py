import math
from typing import Any, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.components.activations import Activation
from xformers.triton.k_activations import get_triton_activation_index
from xformers.triton.k_fused_matmul_bw import fused_matmul_backward
from xformers.triton.k_fused_matmul_fw import fused_matmul

        Compute the derivative with respect to x, other tensors were not trainable inputs.
        