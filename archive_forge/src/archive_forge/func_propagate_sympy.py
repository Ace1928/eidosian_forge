import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
import sympy
from typing_extensions import TypeAlias
import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
def propagate_sympy(self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> IndexPropResult:

    def unwrap(a: Union[Any, IndexPropVar]) -> Any:
        if not isinstance(a, IndexPropVar):
            return a
        return a.value
    new_args = [unwrap(a) for a in args]
    new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
    new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
    is_valid_expr = new_expr is not NotImplemented and (isinstance(new_expr.expr, sympy.Number) or new_expr.expr.is_integer)
    if not is_valid_expr:
        return self.fallback(name, args, kwargs)
    return IndexPropVar.new_symbolic(new_expr)