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
def get_any_has_forward_grad_name(var_names: Tuple[str, ...]) -> str:
    if len(var_names) == 1:
        return f'_any_has_forward_grad_{var_names[0]}'
    else:
        return f'_any_has_forward_grad_{'_'.join(var_names)}'