from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate: gate network
        expert: expert network
        group: group to use for all-to-all communication
    