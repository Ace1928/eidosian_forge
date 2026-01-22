from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def ufunc_type(t: Type, *, binds: ArgName, compute_t: CType) -> NamedCType:
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r
    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, compute_t)
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, compute_t)
    else:
        raise AssertionError(f'unrecognized type {repr(t)}')