from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def get_no_optimization_args(self) -> T.List[str]:
    return ['/Od', '/Oi-']