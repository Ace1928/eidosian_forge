import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
def emit_check_no_requires_grad(tensor_args: List[DifferentiableInput], args_with_derivatives: List[DifferentiableInput]) -> List[str]:
    """Checks that arguments without derivatives don't require grad"""
    body: List[str] = []
    for arg in tensor_args:
        if arg in args_with_derivatives:
            continue
        arg_name = arg.name
        if info and arg_name in info.non_differentiable_arg_names:
            continue
        if arg_name == 'output':
            continue
        body.append(f'check_no_requires_grad({arg_name}, "{arg_name}", "{name}");')
    return body