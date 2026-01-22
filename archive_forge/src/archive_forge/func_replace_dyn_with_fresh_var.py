from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
def replace_dyn_with_fresh_var(self, typ):
    """
        Replace all unknown types with fresh type variables.
        """
    if typ == Dyn:
        new_symbol = Var(next(self.symbol_iter))
        return new_symbol
    elif isinstance(typ, TensorType):
        new_args = [self.replace_dyn_with_fresh_var(a) for a in typ.__args__]
        return TensorType(tuple(new_args))
    elif isinstance(typ, list):
        return [self.replace_dyn_with_fresh_var(t) for t in typ]
    elif isinstance(typ, tuple):
        return (self.replace_dyn_with_fresh_var(t) for t in typ)
    else:
        return typ