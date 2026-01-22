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
def emit_view_lambda(f: NativeFunction, unpacked_bindings: List[Binding]) -> str:
    """Generate an additional lambda function to recover views in backward when as_strided is not supported.
    See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details.
    """
    input_base = 'input_base'
    replay_view_func = ''
    updated_unpacked_args: List[str] = []
    known_view_arg_simple_types: List[CType] = [BaseCType(longT), OptionalCType(BaseCType(longT)), BaseCType(SymIntT), OptionalCType(BaseCType(SymIntT)), BaseCType(boolT), BaseCType(intArrayRefT), BaseCType(symIntArrayRefT), ConstRefCType(BaseCType(tensorT))]
    for unpacked_binding in unpacked_bindings:
        arg, arg_type = (unpacked_binding.name, unpacked_binding.nctype.type)
        if arg == 'self_':
            updated_unpacked_args.append(input_base)
            continue
        if arg_type not in known_view_arg_simple_types:
            known_types_str = ', '.join([str(t) for t in known_view_arg_simple_types])
            raise TypeError(f'You are adding an {arg_type} {arg} argument to op {cpp.name(f.func)} in addition to known types: {known_types_str}. Please update the list or materialize it so that it can be closed over by value, also add a test in pytorch/xla/test/test_operations.py where this code is exercised.')
        if arg_type == BaseCType(intArrayRefT) or arg_type == BaseCType(symIntArrayRefT):
            arg_vec = arg + '_vec'
            replay_view_func += ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec)
            updated_unpacked_args.append(arg_vec)
        elif arg_type == OptionalCType(BaseCType(longT)):
            arg_value = arg + '_val'
            replay_view_func += OPTIONAL_TO_VAL.substitute(arg=arg, val=arg_value, default='0')
            updated_unpacked_args.append(arg_value)
        elif (arg == 'nested_size_' or arg == 'nested_strides_' or arg == 'offsets_') and arg_type == ConstRefCType(BaseCType(tensorT)):
            updated_unpacked_args.append(arg[:-1])
        else:
            updated_unpacked_args.append(arg)
    replay_view_call = emit_view_call(f, input_base, updated_unpacked_args)
    replay_view_func += REPLAY_VIEW_LAMBDA_FUNC.substitute(input_base=input_base, replay_view_call=replay_view_call)
    is_view_with_metadata_change = 'true' if cpp.name(f.func) in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else 'false'
    return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(is_view_with_metadata_change=is_view_with_metadata_change, replay_view_func=replay_view_func)