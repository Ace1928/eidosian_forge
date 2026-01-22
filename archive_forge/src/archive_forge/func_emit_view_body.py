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
def emit_view_body(fn: NativeFunctionWithDifferentiabilityInfo, var: str) -> Tuple[str, str]:
    f = fn.func
    base_name = get_base_name(f)
    view_info = get_view_info(f)
    call = ''
    differentiable_outputs = gen_differentiable_outputs(fn)
    differentiable_output_vars = {r.name for r in differentiable_outputs}
    if not isinstance(view_info, str):
        raise TypeError(f'The view info should be a string for {base_name}, but it is: {view_info}')
    if len(differentiable_output_vars) == 0:
        rhs_value = f'as_view({view_info}, {var}, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false)'
    elif len(differentiable_output_vars) == 1:
        return_info = differentiable_outputs[0]
        if not is_tensor_type(return_info.type) and (not is_tensor_list_type(return_info.type)):
            raise RuntimeError(f'{base_name} that return differentiable views can only return Tensor or Tensor[]')

        def get_creation_meta_in_mode(original: str) -> str:
            creation_meta_with_grad_mode = f'(at::GradMode::is_enabled() ? {original} : CreationMeta::NO_GRAD_MODE)'
            return f'InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : {creation_meta_with_grad_mode}'
        if is_tensor_list_type(return_info.type):
            creation_meta = get_creation_meta_in_mode('CreationMeta::MULTI_OUTPUT_NODE')
            call += f'as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ {creation_meta});'
            rhs_value = f'std::move({var})'
        else:
            _, unpacked_bindings = unpack_args(f)
            call += emit_view_lambda(f, unpacked_bindings)
            creation_meta = get_creation_meta_in_mode('CreationMeta::DEFAULT')
            rhs_value = f'as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ {creation_meta})'
    else:
        raise RuntimeError('Function that return multiple differentiable output when at least one of them is view is not supported.')
    return (call, rhs_value)