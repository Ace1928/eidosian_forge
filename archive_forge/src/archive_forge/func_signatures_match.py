import torch
from torch import Tensor
import inspect
import warnings
from typing import Dict, List, Optional, Set
from torch.types import Number
def signatures_match(decomposition_sig, torch_op_sig):
    decomp_params = decomposition_sig.parameters
    op_params = torch_op_sig.parameters
    if len(decomp_params) != len(op_params):
        return False
    for decomp_param, op_param in zip(decomp_params.values(), op_params.values()):
        inspect_empty = inspect._empty
        for field in ['name', 'annotation']:
            if field == 'name' and decomp_param.name == 'self':
                warnings.warn("PyTorch uses 'input' instead of 'self' on public api")
            if getattr(decomp_param, field) != getattr(op_param, field):
                return False
        decomp_default = decomp_param.default
        op_default = op_param.default
        if decomp_default != inspect_empty and op_default != inspect_empty:
            if decomp_default != op_default:
                return False
    return decomposition_sig.return_annotation == torch_op_sig.return_annotation