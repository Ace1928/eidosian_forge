from dataclasses import dataclass
from typing import List, Optional, Set
import torchgen.api.cpp as aten_cpp
from torchgen.api.types import Binding, CType
from torchgen.model import FunctionSchema, NativeFunction
from .types import contextArg
from torchgen.executorch.api import et_cpp
@staticmethod
def from_native_function(f: NativeFunction, *, prefix: str='') -> 'ExecutorchCppSignature':
    return ExecutorchCppSignature(func=f.func, prefix=prefix, cpp_no_default_args=f.cpp_no_default_args)