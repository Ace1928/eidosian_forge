import torch
from torch import Tensor
import inspect
import warnings
from typing import Dict, List, Optional, Set
from torch.types import Number
def decomposition_decorator(f):
    nonlocal registry
    if registry is None:
        registry = decomposition_table
    assert isinstance(aten_op, torch._ops.OpOverload)
    assert f.__name__ not in function_name_set, f'Duplicated function name {f.__name__}'
    function_name_set.add(f.__name__)
    scripted_func = torch.jit.script(f)
    torch._C._jit_pass_inline(scripted_func.graph)
    for _ in range(2):
        torch._C._jit_pass_peephole(scripted_func.graph)
        torch._C._jit_pass_constant_propagation(scripted_func.graph)
    registry[str(aten_op._schema)] = scripted_func
    return f