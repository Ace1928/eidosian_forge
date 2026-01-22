import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
def gen_registration_headers(backend_index: BackendIndex, per_operator_headers: bool, rocm: bool) -> List[str]:
    if per_operator_headers:
        headers = ['#include <ATen/ops/as_strided_native.h>']
    else:
        headers = ['#include <ATen/NativeFunctions.h>']
    if backend_index.dispatch_key in (DispatchKey.CPU, DispatchKey.Meta):
        headers.append('#include <ATen/EmptyTensor.h>')
    elif backend_index.dispatch_key == DispatchKey.CUDA:
        if rocm:
            headers.append('#include <ATen/hip/EmptyTensor.h>')
        else:
            headers.append('#include <ATen/cuda/EmptyTensor.h>')
    elif backend_index.dispatch_key == DispatchKey.MPS:
        headers.append('#include <ATen/mps/EmptyTensor.h>')
    elif per_operator_headers:
        headers += ['#include <ATen/ops/empty.h>', '#include <ATen/ops/empty_strided.h>', '#include <ATen/ops/_copy_from_and_resize.h>', '#include <ATen/ops/_copy_from.h>']
    else:
        headers.append('#include <ATen/Functions.h>')
    return headers