from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def ufunctor_apply_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    return Binding(nctype=ufunctor_apply_type(a.type, binds=a.name, scalar_t=scalar_t), name=a.name, default=None, argument=a)