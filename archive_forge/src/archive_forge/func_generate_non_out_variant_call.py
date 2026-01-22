import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def generate_non_out_variant_call(g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    schema = g.functional.func
    assert not schema.is_out_fn()
    kernel_name = get_kernel_name(g, backend_index)
    arg_names = (arg.name for arg in schema.schema_order_arguments())
    namespace_name = 'cpu' if g.structured else 'native'
    return f'at::{namespace_name}::{kernel_name}({','.join(arg_names)})'