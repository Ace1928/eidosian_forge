import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
import sympy
from typing_extensions import TypeAlias
import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any:
    if isinstance(expr, sympy.Integer):
        return self._inner.constant(int(expr), dtype)
    elif expr.is_number:
        return self._inner.constant(float(expr), dtype)
    return self._inner.index_expr(expr, dtype)