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
def make_simplify_loops_cache(self):
    """
        self._simplify_with_ranges() can be expensive, cache its results
        """
    cache: Dict[Tuple[Any, ...], Any] = dict()
    replacement_count = len(self.replacements)

    def simplify_loops(index_vars, sizes, index_formulas):
        nonlocal replacement_count
        if replacement_count != len(self.replacements):
            cache.clear()
            replacement_count = len(self.replacements)
        key = (*index_vars, *sizes, *index_formulas)
        result = cache.get(key, None)
        if result is None:
            result = self._simplify_loops_impl(index_vars, sizes, index_formulas)
            cache[key] = result
        return result
    return simplify_loops