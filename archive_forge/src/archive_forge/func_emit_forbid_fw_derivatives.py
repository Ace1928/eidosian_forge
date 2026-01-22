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
def emit_forbid_fw_derivatives(is_out_fn: bool=False) -> str:
    if is_out_fn:
        msg = 'because it is an out= function'
    else:
        msg = 'because it has not been implemented yet.\\nPlease file an issue to PyTorch at https://github.com/pytorch/pytorch/issues/new?template=feature-request.yml so that we can prioritize its implementation.'
    cond = get_any_has_fw_grad_cond(derivative=None)
    return FW_DERIVATIVE_FORBID_TEMPLATE.substitute(cond=cond, name=name, msg=msg) if cond != '' else ''