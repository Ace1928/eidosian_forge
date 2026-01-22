from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
@contextlib.contextmanager
def cached_compile(self, code: 'mesonlib.FileOrString', cdata: coredata.CoreData, *, extra_args: T.Union[None, T.List[str], CompilerArgs]=None, mode: CompileCheckMode=CompileCheckMode.LINK, temp_dir: T.Optional[str]=None) -> T.Iterator[CompileResult]:
    textra_args: T.Tuple[str, ...] = tuple(extra_args) if extra_args is not None else tuple()
    key: coredata.CompilerCheckCacheKey = (tuple(self.exelist), self.version, code, textra_args, mode)
    if key in cdata.compiler_check_cache:
        p = cdata.compiler_check_cache[key]
        p.cached = True
        mlog.debug('Using cached compile:')
        mlog.debug('Cached command line: ', ' '.join(p.command), '\n')
        mlog.debug('Code:\n', code)
        mlog.debug('Cached compiler stdout:\n', p.stdout)
        mlog.debug('Cached compiler stderr:\n', p.stderr)
        yield p
    else:
        with self.compile(code, extra_args=extra_args, mode=mode, want_output=False, temp_dir=temp_dir) as p:
            cdata.compiler_check_cache[key] = p
            yield p