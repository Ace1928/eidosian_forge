from __future__ import annotations
from pathlib import Path
from .traceparser import CMakeTraceParser
from ..envconfig import CMakeSkipCompilerTest
from .common import language_map, cmake_get_generator_args
from .. import mlog
import shutil
import typing as T
from enum import Enum
from textwrap import dedent
def generate_cache(self) -> str:
    if not self.skip_check:
        return ''
    res = ''
    for name, v in self.cmakestate.cmake_cache.items():
        res += f'{name}:{v.type}={';'.join(v.value)}\n'
    return res