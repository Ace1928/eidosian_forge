import functools
import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy
from .utils import sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
def make_stride_vars_cache(self):
    cache = self._lru_cache(self._stride_vars)

    def stride_vars(index: Expr, vars: List[sympy.Symbol], support_vars: Optional[List[sympy.Symbol]]=None) -> List[Expr]:
        if not support_vars:
            support_vars = vars
        return cache(index, tuple(vars), tuple(support_vars))
    return stride_vars