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
@with_native_function_with_differentiability_info
def emit_inplace_or_view_body(fn: NativeFunctionWithDifferentiabilityInfo) -> List[str]:
    f = fn.func
    inplace_view_body: List[str] = []
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()
    dispatch_key_set = 'ks & c10::after_ADInplaceOrView_keyset'
    redispatch_args = ', '.join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])
    if modifies_arguments(f):
        inplace_view_body.append(INPLACE_REDISPATCH.substitute(unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=redispatch_args))
        for r in cpp.return_names(f):
            inplace_view_body.append(f'increment_version({r});')
    else:
        assert get_view_info(f) is not None
        inplace_view_body.append(VIEW_REDISPATCH.substitute(assign_return_values='auto ' + TMP_VAR + ' = ', unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=redispatch_args))
        call, rhs_value = emit_view_body(fn, TMP_VAR)
        inplace_view_body.append(call)
        assert rhs_value is not None
        inplace_view_body.append(ASSIGN_RETURN_VALUE.substitute(return_values=tie_return_values(f), rhs_value=rhs_value))
    if f.func.returns:
        inplace_view_body.append(f'return {get_return_value(f)};')
    return inplace_view_body