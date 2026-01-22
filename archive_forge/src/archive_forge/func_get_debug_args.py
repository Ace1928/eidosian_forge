from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
def get_debug_args(self, is_debug: bool) -> T.List[str]:
    return self.DEBUG_ARGS[is_debug]