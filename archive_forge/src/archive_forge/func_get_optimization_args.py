from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
def get_optimization_args(self, optimization_level: str) -> T.List[str]:
    return self.OPTIM_ARGS[optimization_level]