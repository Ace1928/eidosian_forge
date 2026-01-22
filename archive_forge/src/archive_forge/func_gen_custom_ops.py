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
def gen_custom_ops(*, native_functions: Sequence[NativeFunction], selector: SelectiveBuilder, kernel_index: ETKernelIndex, cpu_fm: FileManager, rocm: bool) -> None:
    dispatch_key = DispatchKey.CPU
    anonymous_definition, static_init_dispatch_registrations = gen_custom_ops_registration(native_functions=native_functions, selector=selector, kernel_index=kernel_index, rocm=rocm)
    cpu_fm.write_with_template(f'Register{dispatch_key}CustomOps.cpp', 'RegisterDispatchKeyCustomOps.cpp', lambda: {'ops_headers': '#include "CustomOpsNativeFunctions.h"', 'DispatchKey': dispatch_key, 'dispatch_namespace': dispatch_key.lower(), 'dispatch_namespaced_definitions': '', 'dispatch_anonymous_definitions': anonymous_definition, 'static_init_dispatch_registrations': static_init_dispatch_registrations})
    cpu_fm.write_with_template(f'Register{dispatch_key}Stub.cpp', 'RegisterDispatchKeyCustomOps.cpp', lambda: {'ops_headers': '', 'DispatchKey': dispatch_key, 'dispatch_namespace': dispatch_key.lower(), 'dispatch_namespaced_definitions': '', 'dispatch_anonymous_definitions': list(mapMaybe(ComputeNativeFunctionStub(), native_functions)), 'static_init_dispatch_registrations': static_init_dispatch_registrations})
    aten_schema_registrations, schema_registrations = get_native_function_schema_registrations(native_functions=native_functions, schema_selector=selector)
    cpu_fm.write('RegisterSchema.cpp', lambda: {'schema_registrations': schema_registrations, 'aten_schema_registrations': aten_schema_registrations})