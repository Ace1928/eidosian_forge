from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def unwrap_tensor_args(sig: DispatcherSignature, *, is_view_op: bool) -> Tuple[str, List[Binding]]:
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            unwrapped_name = f'{arg.name}_'
            maybe_sync_input = '' if is_view_op else f'at::functionalization::impl::sync({arg.name});'
            unwrapped_type, conversion_fn = get_owning_type(arg.nctype.remove_const_ref().type)
            unwrapped_tensor_args.append(f'\n      {unwrapped_type.cpp_type()} {unwrapped_name};\n      if (at::functionalization::impl::isFunctionalTensor({arg.name})) {{\n        {maybe_sync_input}\n        {unwrapped_name} = at::functionalization::impl::from_functional_tensor({arg.name});\n      }} else {{\n        {unwrapped_name} = {conversion_fn(arg.name)};\n      }}')
            context.append(arg.with_name(unwrapped_name))
        else:
            context.append(arg)
    unwrap_tensor_args_str = '\n      '.join(unwrapped_tensor_args)
    return (unwrap_tensor_args_str, context)