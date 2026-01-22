import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def generate_out_variant_call(g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    schema = g.out.func
    assert schema.is_out_fn()
    arg_names = []
    kernel_name = get_out_kernel_name(g, backend_index)
    if g.structured:
        arg_names = [out_arg.name for out_arg in schema.arguments.out]
    else:
        arg_names = []
    for arg in schema.arguments.non_out:
        if isinstance(arg, SelfArgument):
            arg_names.append(arg.argument.name)
        else:
            assert isinstance(arg, Argument)
            arg_names.append(arg.name)
    if not g.structured:
        assert len(schema.arguments.out) == 1
        arg_names.append(schema.arguments.out[0].name)
    cpp_arg_names = ','.join(arg_names)
    namespace_name = 'cpu' if g.structured else 'native'
    return f'at::{namespace_name}::{kernel_name}({cpp_arg_names})'