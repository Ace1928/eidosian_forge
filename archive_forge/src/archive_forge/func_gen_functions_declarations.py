import argparse
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, Union
import yaml
from torchgen import dest
from torchgen.api import cpp as aten_cpp
from torchgen.api.types import CppSignature, CppSignatureGroup, CType, NamedCType
from torchgen.context import (
from torchgen.executorch.api import et_cpp
from torchgen.executorch.api.custom_ops import (
from torchgen.executorch.api.types import contextArg, ExecutorchCppSignature
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.executorch.model import ETKernelIndex, ETKernelKey, ETParsedYaml
from torchgen.executorch.parse import ET_FIELDS, parse_et_yaml, parse_et_yaml_struct
from torchgen.gen import (
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
def gen_functions_declarations(*, native_functions: Sequence[NativeFunction], kernel_index: ETKernelIndex, selector: SelectiveBuilder, use_aten_lib: bool, custom_ops_native_functions: Optional[Sequence[NativeFunction]]=None) -> str:
    """
    Generates namespace separated C++ function API inline declaration/definitions.
    Native functions are grouped by namespaces and the generated code is wrapped inside
    namespace blocks.

    E.g., for `custom_1::foo.out` in yaml file we will generate a C++ API as a symbol
    in `torch::executor::custom_1::foo_out`. This way we avoid symbol conflict when
    the other `custom_2::foo.out` is available.
    """
    dispatch_key = DispatchKey.CPU
    backend_index = kernel_index._to_backend_index()
    ns_grouped_functions = defaultdict(list)
    for native_function in native_functions:
        ns_grouped_functions[native_function.namespace].append(native_function)
    functions_declarations = ''
    newline = '\n'
    for namespace in ns_grouped_functions:
        ns_helper = NamespaceHelper(namespace_str=namespace, entity_name='', max_level=3)
        declarations = list(mapMaybe(ComputeFunction(static_dispatch_backend_indices=[backend_index], selector=selector, use_aten_lib=use_aten_lib, is_custom_op=lambda f: custom_ops_native_functions is not None and f in custom_ops_native_functions), ns_grouped_functions[namespace]))
        functions_declarations += f'\n{ns_helper.prologue}\n{newline.join(declarations)}\n{ns_helper.epilogue}\n        '
    return functions_declarations