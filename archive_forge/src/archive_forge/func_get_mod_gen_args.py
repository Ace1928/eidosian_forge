from __future__ import annotations
import subprocess, os.path
import typing as T
from ..mesonlib import EnvironmentException
from .compilers import Compiler, clike_debug_args
def get_mod_gen_args(self) -> T.List[str]:
    return ['-emit-module']