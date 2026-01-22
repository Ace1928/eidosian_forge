from __future__ import annotations
import functools
import subprocess, os.path
import textwrap
import re
import typing as T
from .. import coredata
from ..mesonlib import EnvironmentException, MesonException, Popen_safe_logged, OptionKey
from .compilers import Compiler, clike_debug_args
def get_sysroot(self) -> str:
    cmd = self.get_exelist(ccache=False) + ['--print', 'sysroot']
    p, stdo, stde = Popen_safe_logged(cmd)
    return stdo.split('\n', maxsplit=1)[0]