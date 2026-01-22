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
def emit_any_requires_grad() -> List[str]:
    extra_condition = ''
    if info and info.output_differentiability_conditions:
        assert len(info.output_differentiability_conditions) == 1
        extra_condition = f'_any_requires_grad &= ({info.output_differentiability_conditions[0]});'
    names_of_args_with_derivatives = [arg.name for arg in args_with_derivatives]
    if is_inplace_foreach and info is not None:
        for i, arg in enumerate(names_of_args_with_derivatives):
            for f_arg, r_arg in inplace_foreacharg2refarg.items():
                if arg == r_arg.name:
                    names_of_args_with_derivatives[i] = f_arg.name
    return [SETUP_ANY_REQUIRES_GRAD.substitute(args_with_derivatives=names_of_args_with_derivatives, extra_differentiability_conditions=extra_condition)]