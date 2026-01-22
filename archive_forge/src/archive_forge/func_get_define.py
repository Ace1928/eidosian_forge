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
def get_define(self, dname: str, prefix: str, env: 'Environment', extra_args: T.Union[T.List[str], T.Callable[[CompileCheckMode], T.List[str]]], dependencies: T.Optional[T.List['Dependency']], disable_cache: bool=False) -> T.Tuple[str, bool]:
    delim_start = '"MESON_GET_DEFINE_DELIMITER_START"\n'
    delim_end = '\n"MESON_GET_DEFINE_DELIMITER_END"'
    sentinel_undef = '"MESON_GET_DEFINE_UNDEFINED_SENTINEL"'
    code = f'\n        {prefix}\n        #ifndef {dname}\n        # define {dname} {sentinel_undef}\n        #endif\n        {delim_start}{dname}{delim_end}'
    args = self.build_wrapper_args(env, extra_args, dependencies, mode=CompileCheckMode.PREPROCESS).to_native()
    func = functools.partial(self.cached_compile, code, env.coredata, extra_args=args, mode=CompileCheckMode.PREPROCESS)
    if disable_cache:
        func = functools.partial(self.compile, code, extra_args=args, mode=CompileCheckMode.PREPROCESS)
    with func() as p:
        cached = p.cached
        if p.returncode != 0:
            raise mesonlib.EnvironmentException(f'Could not get define {dname!r}')
    star_idx = p.stdout.find(delim_start)
    end_idx = p.stdout.rfind(delim_end)
    if star_idx == -1 or end_idx == -1 or star_idx == end_idx:
        raise mesonlib.MesonBugException('Delimiters not found in preprocessor output.')
    define_value = p.stdout[star_idx + len(delim_start):end_idx]
    if define_value == sentinel_undef:
        define_value = None
    else:
        define_value = self._concatenate_string_literals(define_value).strip()
    return (define_value, cached)