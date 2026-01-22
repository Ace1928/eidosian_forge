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
def cross_sizeof(self, typename: str, prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> int:
    if extra_args is None:
        extra_args = []
    t = f'{prefix}\n        #include <stddef.h>\n        int main(void) {{\n            {typename} something;\n            return 0;\n        }}'
    if not self.compiles(t, env, extra_args=extra_args, dependencies=dependencies)[0]:
        return -1
    return self.cross_compute_int(f'sizeof({typename})', None, None, None, prefix, env, extra_args, dependencies)