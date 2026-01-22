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
def has_members(self, typename: str, membernames: T.List[str], prefix: str, env: 'Environment', *, extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[bool, bool]:
    if extra_args is None:
        extra_args = []
    members = ''.join((f'foo.{member};\n' for member in membernames))
    t = f'{prefix}\n        void bar(void) {{\n            {typename} foo;\n            {members}\n        }}'
    return self.compiles(t, env, extra_args=extra_args, dependencies=dependencies)