from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def ufunc_argument(a: Argument, compute_t: CType) -> Binding:
    return Binding(nctype=ufunc_type(a.type, binds=a.name, compute_t=compute_t), name=a.name, default=None, argument=a)