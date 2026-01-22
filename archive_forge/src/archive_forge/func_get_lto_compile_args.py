from __future__ import annotations
import abc
import functools
import os
import multiprocessing
import pathlib
import re
import subprocess
import typing as T
from ... import mesonlib
from ... import mlog
from ...mesonlib import OptionKey
from mesonbuild.compilers.compilers import CompileCheckMode
def get_lto_compile_args(self, *, threads: int=0, mode: str='default') -> T.List[str]:
    if threads == 0:
        if mesonlib.version_compare(self.version, '>= 10.0'):
            return ['-flto=auto']
        return [f'-flto={multiprocessing.cpu_count()}']
    elif threads > 0:
        return [f'-flto={threads}']
    return super().get_lto_compile_args(threads=threads)