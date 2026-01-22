from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def make_sig(*, faithful: bool, symint: bool) -> CppSignature:
    return CppSignature(func=func, faithful=faithful, symint=symint, method=method, fallback_binding=fallback_binding, cpp_no_default_args=f.cpp_no_default_args)