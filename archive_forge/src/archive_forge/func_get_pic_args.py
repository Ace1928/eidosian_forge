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
def get_pic_args(self) -> T.List[str]:
    if self.info.is_windows() or self.info.is_cygwin() or self.info.is_darwin():
        return []
    return ['-fPIC']