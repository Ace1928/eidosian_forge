from dataclasses import dataclass
from typing import List, Optional, Set
import torchgen.api.cpp as aten_cpp
from torchgen.api.types import Binding, CType
from torchgen.model import FunctionSchema, NativeFunction
from .types import contextArg
from torchgen.executorch.api import et_cpp

    This signature is merely a CppSignature with Executorch types (optionally
    contains KernelRuntimeContext as well). The inline definition of
    CppSignature is generated in Functions.h and it's used by unboxing
    functions.
    