from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def get_compile_debugfile_args(self, rel_obj: str, pch: bool=False) -> T.List[str]:
    args = super().get_compile_debugfile_args(rel_obj, pch)
    if pch and mesonlib.version_compare(self.version, '>=18.0'):
        args = ['/FS'] + args
    return args