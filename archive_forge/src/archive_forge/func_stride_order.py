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
def stride_order(self, index: Expr, vars: List[sympy.Symbol]) -> List[int]:
    strides = tuple(map(abs, self.stride_hints(index, vars)))
    order = list(range(len(strides)))
    order.sort(key=lambda x: (strides[x] == 0, strides[x]))
    return order