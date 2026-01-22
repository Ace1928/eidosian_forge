import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def generate_call_to_view_ops(g: NativeFunctionsViewGroup, backend_index: BackendIndex) -> str:
    schema = g.view.func
    kernel_name = cpp.name(schema)
    kernel = backend_index.get_kernel(g.view)
    if kernel:
        kernel_name = kernel.kernel
    arg_names = (arg.name for arg in schema.schema_order_arguments())
    namespace_name = 'native'
    return f'at::{namespace_name}::{kernel_name}({','.join(arg_names)})'