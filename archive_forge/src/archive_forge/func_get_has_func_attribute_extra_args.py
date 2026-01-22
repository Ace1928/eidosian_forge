from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
def get_has_func_attribute_extra_args(self, name: str) -> T.List[str]:
    return ['-diag-error', '1292']