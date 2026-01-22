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
def get_compiler_dirs(self, env: 'Environment', name: str) -> T.List[str]:
    """
        Get dirs from the compiler, either `libraries:` or `programs:`
        """
    stdo = self._get_search_dirs(env)
    for line in stdo.split('\n'):
        if line.startswith(name + ':'):
            return self._split_fetch_real_dirs(line.split('=', 1)[1])
    return []