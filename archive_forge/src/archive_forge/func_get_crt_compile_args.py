from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def get_crt_compile_args(self, crt_val: str, buildtype: str) -> T.List[str]:
    crt_val = self.get_crt_val(crt_val, buildtype)
    return self.crt_args[crt_val]