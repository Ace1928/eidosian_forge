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
def get_any_has_fw_grad_cond(derivative: Optional[ForwardDerivative]) -> str:
    if derivative is None:
        to_check: List[str] = []
        for inp in list(mapMaybe(gen_differentiable_input, f.func.arguments.non_out + list(f.func.arguments.out))):
            if is_tensor_type(inp.type):
                to_check.append(FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp.name))
            elif is_tensor_list_type(inp.type):
                to_check.append(FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE.substitute(req_inp=inp.name))
            else:
                raise RuntimeError(f'Unsupported input type for "{name}" when forbidding forward AD usage.')
        return f'({' || '.join(to_check)})'
    else:
        assert derivative.required_inputs_fw_grad is not None
        if len(derivative.required_inputs_fw_grad) == 0:
            if not (len(differentiable_inputs) == 1 and is_tensor_list_type(differentiable_inputs[0].type)):
                raise RuntimeError(f'No differentiable input to "{name}" is a differentiable Tensor (as the provided forward AD formula does not use any input tangent) even though a forward gradient formula has been defined for it. This case should only happen for function that take a single TensorList as input. All other cases are not supported right now.')
            any_has_fw_grad = 'true'
        else:
            any_has_fw_grad = ' || '.join([(FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE if is_tensor_list_type(inp.type) else FW_DERIVATIVE_CHECK_TEMPLATE).substitute(req_inp=inp.name) for inp in differentiable_inputs if inp.name in derivative.required_inputs_fw_grad])
            any_has_fw_grad = f'({any_has_fw_grad})'
        return any_has_fw_grad