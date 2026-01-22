from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def stub_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    return [r for a in g.out.func.arguments.flat_non_out if not a.type.is_tensor_like() for r in structured.argument(a)]