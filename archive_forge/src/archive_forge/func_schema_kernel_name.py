from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def schema_kernel_name(func: FunctionSchema, dispatch_key: DispatchKey) -> str:
    assert func.is_out_fn(), 'ufunc.kernel_name should only be invoked on out schemas'
    return f'ufunc_{func.name.name}_{dispatch_key}'