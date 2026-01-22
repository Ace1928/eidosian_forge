from functools import partial
from typing import Optional
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup
from flash_attn.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from flash_attn.utils.distributed import (
def fused_mlp_func(x: Tensor, weight1: Tensor, weight2: Tensor, bias1: Optional[Tensor]=None, bias2: Optional[Tensor]=None, activation: str='gelu_approx', save_pre_act: bool=True, return_residual: bool=False, checkpoint_lvl: int=0, heuristic: int=0, process_group: Optional[ProcessGroup]=None, sequence_parallel: bool=True):
    assert activation in ['gelu_approx', 'relu', 'sqrelu']
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (x.dtype == torch.float32 and torch.is_autocast_enabled())
    dim_eligible = not save_pre_act or x.shape[-1] % (128 if activation == 'relu' else 8) == 0
    if x.is_cuda and weight1.is_cuda and weight2.is_cuda and (bias1 is None or bias1.is_cuda) and (bias2 is None or bias2.is_cuda) and dtype_eligible and dim_eligible:
        return FusedMLPFunc.apply(x, weight1, bias1, weight2, bias2, activation, save_pre_act, return_residual, checkpoint_lvl, heuristic, process_group, sequence_parallel)
    else:
        assert process_group is None
        pre_act = F.linear(x, weight1, bias1)
        activation_fn = partial(F.gelu, approximate='tanh') if activation == 'gelu_approx' else partial(F.relu, inplace=True)
        output1 = activation_fn(pre_act)
        output2 = F.linear(output1, weight2, bias2)
        return output2 if not return_residual else (output2, x)