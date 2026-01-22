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
def get_warn_args(self, level: str) -> T.List[str]:
    args = super().get_warn_args(level)
    if mesonlib.version_compare(self.version, '<4.8.0') and '-Wpedantic' in args:
        args[args.index('-Wpedantic')] = '-pedantic'
    return args