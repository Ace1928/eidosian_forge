import functools
import json
import os
from typing import Any, Dict, Optional, Tuple
import torch
import triton
import triton.language as tl
from vllm._C import ops
from vllm.logger import init_logger
from vllm.utils import is_hip

    This function computes a Mixture of Experts (MoE) layer using two sets of weights, w1 and w2, and top-k gating mechanism.
    
    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place. Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override for the kernel configuration.
    
    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    