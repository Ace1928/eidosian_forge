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
def convert_to_sympy_symbols(self, typ):
    """
        Replace all unknown types with fresh type variables.
        """
    if isinstance(typ, Var):
        return sympy.symbols(str(typ))
    elif isinstance(typ, TensorType):
        new_args = [self.convert_to_sympy_symbols(a) for a in typ.__args__]
        return TensorType(tuple(new_args))
    elif isinstance(typ, list):
        return [self.convert_to_sympy_symbols(t) for t in typ]
    elif isinstance(typ, tuple):
        return (self.convert_to_sympy_symbols(t) for t in typ)
    else:
        return typ