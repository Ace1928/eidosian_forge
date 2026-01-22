from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def ufunc_arguments(g: NativeFunctionsGroup, *, compute_t: CType) -> List[Binding]:
    return [ufunc_argument(a, compute_t=compute_t) for a in g.functional.func.arguments.flat_non_out]