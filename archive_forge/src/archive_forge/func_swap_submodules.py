import operator
from functools import partial
from typing import Any, Callable, Dict
from sympy import Expr
import torch
from torch.utils._sympy.value_ranges import bound_sympy, ValueRangeAnalysis, ValueRanges
from .ir import InterpreterShim, LoopBody, LoopBodyBlock
from .utils import cache_on_self, dominated_nodes
from .virtualized import V
def swap_submodules(self, submodules: Dict[str, Callable[..., Any]]) -> Dict[str, Callable[..., ValueRanges]]:
    result: Dict[str, Callable[..., ValueRanges]] = {}
    for key in submodules.keys():
        if key == 'get_index':
            result[key] = self.get_index
        elif 'masked_subblock' in key:
            subblock = self.loop_body.subblocks[key]

            def make_fn(subblock):
                return lambda mask, value: self.masked_subblock(subblock, self._bounds, mask, value, result)
            result[key] = make_fn(subblock)
        else:
            assert 'set_indirect' in key
            idx = int(key[len('set_indirect'):])
            var = self.loop_body.indirect_vars[idx]
            indirect = partial(self.set_indirect, var)
            result[key] = indirect
    return result