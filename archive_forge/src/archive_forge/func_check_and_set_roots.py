from __future__ import annotations
import re
import dataclasses
import functools
import typing as T
from pathlib import Path
from .. import mlog
from .. import mesonlib
from .base import DependencyException, SystemDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .misc import threads_factory
def check_and_set_roots(self, roots: T.List[Path], use_system: bool) -> None:
    roots = list(mesonlib.OrderedSet(roots))
    for j in roots:
        mlog.debug(f'Checking potential boost root {j.as_posix()}')
        inc_dirs = self.detect_inc_dirs(j)
        inc_dirs = sorted(inc_dirs, reverse=True)
        if not inc_dirs:
            continue
        lib_dirs = self.detect_lib_dirs(j, use_system)
        self.is_found = self.run_check(inc_dirs, lib_dirs)
        if self.is_found:
            self.boost_root = j
            break