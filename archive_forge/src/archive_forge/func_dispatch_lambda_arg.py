from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def dispatch_lambda_arg(cpp_arg: Binding) -> DispatchLambdaArgument:
    type_str = cpp_arg.type
    is_out_arg = cpp_arg.name in out_args
    if ps.method and cpp_arg.name == 'self':
        type_str = 'const at::Tensor &'
    else:
        ensure_temp_safe = len(out_args) <= 1 or not is_out_arg
        if ensure_temp_safe:
            type_str = {'at::Tensor &': 'at::Tensor'}.get(type_str, type_str)
    return DispatchLambdaArgument(name=cpp_arg.name, type_str=type_str, is_out_arg=is_out_arg)