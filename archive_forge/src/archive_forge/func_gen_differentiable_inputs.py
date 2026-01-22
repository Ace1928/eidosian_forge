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
@with_native_function
def gen_differentiable_inputs(f: NativeFunction) -> List[DifferentiableInput]:
    arguments = list(f.func.arguments.non_out)
    if is_inplace_foreach and info is not None:
        for i, arg in enumerate(f.func.arguments.flat_non_out):
            if arg in inplace_foreacharg2refarg:
                mapped_arg = inplace_foreacharg2refarg[arg]
                arguments[i] = Argument(mapped_arg.name, mapped_arg.type, mapped_arg.default, mapped_arg.annotation)
    return list(mapMaybe(gen_differentiable_input, arguments))