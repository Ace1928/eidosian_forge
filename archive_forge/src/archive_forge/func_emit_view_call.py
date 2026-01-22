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
def emit_view_call(f: NativeFunction, input_base: str, unpacked_args: Sequence[str]) -> str:
    return CALL_DISPATCH.substitute(unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=unpacked_args)