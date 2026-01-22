from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
def get_xcompiler_val(flag: str, flagit: T.Iterator[str]) -> str:
    if is_xcompiler_flag_glued(flag):
        return flag[len('-Xcompiler='):]
    else:
        try:
            return next(flagit)
        except StopIteration:
            return ''