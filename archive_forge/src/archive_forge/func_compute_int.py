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
def compute_int(self, expression: str, low: T.Optional[int], high: T.Optional[int], guess: T.Optional[int], prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]], dependencies: T.Optional[T.List['Dependency']]=None) -> int:
    if extra_args is None:
        extra_args = []
    if self.is_cross:
        return self.cross_compute_int(expression, low, high, guess, prefix, env, extra_args, dependencies)
    t = f'{prefix}\n        #include<stddef.h>\n        #include<stdio.h>\n        int main(void) {{\n            printf("%ld\\n", (long)({expression}));\n            return 0;\n        }}'
    res = self.run(t, env, extra_args=extra_args, dependencies=dependencies)
    if not res.compiled:
        return -1
    if res.returncode != 0:
        raise mesonlib.EnvironmentException('Could not run compute_int test binary.')
    return int(res.stdout)