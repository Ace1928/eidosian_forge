from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def linker_to_compiler_args(self, args: T.List[str]) -> T.List[str]:
    return ['/link'] + args