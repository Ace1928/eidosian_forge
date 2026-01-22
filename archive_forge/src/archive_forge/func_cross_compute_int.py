from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def cross_compute_int(self, expression: str, low: T.Optional[int], high: T.Optional[int], guess: T.Optional[int], prefix: str, env: 'Environment', extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> int:
    if isinstance(guess, int):
        if self._compile_int(f'{expression} == {guess}', prefix, env, extra_args, dependencies):
            return guess
    maxint = 2147483647
    minint = -2147483648
    if not isinstance(low, int) or not isinstance(high, int):
        if self._compile_int(f'{expression} >= 0', prefix, env, extra_args, dependencies):
            low = cur = 0
            while self._compile_int(f'{expression} > {cur}', prefix, env, extra_args, dependencies):
                low = cur + 1
                if low > maxint:
                    raise mesonlib.EnvironmentException('Cross-compile check overflowed')
                cur = min(cur * 2 + 1, maxint)
            high = cur
        else:
            high = cur = -1
            while self._compile_int(f'{expression} < {cur}', prefix, env, extra_args, dependencies):
                high = cur - 1
                if high < minint:
                    raise mesonlib.EnvironmentException('Cross-compile check overflowed')
                cur = max(cur * 2, minint)
            low = cur
    else:
        if high < low:
            raise mesonlib.EnvironmentException('high limit smaller than low limit')
        condition = f'{expression} <= {high} && {expression} >= {low}'
        if not self._compile_int(condition, prefix, env, extra_args, dependencies):
            raise mesonlib.EnvironmentException('Value out of given range')
    while low != high:
        cur = low + int((high - low) / 2)
        if self._compile_int(f'{expression} <= {cur}', prefix, env, extra_args, dependencies):
            high = cur
        else:
            low = cur + 1
    return low