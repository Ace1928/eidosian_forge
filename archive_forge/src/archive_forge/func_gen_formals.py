from typing import Dict, List, Optional, Sequence, Tuple
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import FileManager
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
@with_native_function
def gen_formals(f: NativeFunction) -> str:
    return ', '.join(['c10::DispatchKeySet ks'] + [f'{cpp.argument_type(a, binds='__placeholder__', symint=True).cpp_type()} {a.name}' for a in f.func.schema_order_arguments()])