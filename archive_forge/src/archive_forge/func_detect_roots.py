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
def detect_roots(self) -> None:
    roots: T.List[Path] = []
    try:
        boost_pc = PkgConfigDependency('boost', self.env, {'required': False})
        if boost_pc.found():
            boost_root = boost_pc.get_variable(pkgconfig='prefix')
            if boost_root:
                roots += [Path(boost_root)]
    except DependencyException:
        pass
    inc_paths = [Path(x) for x in self.clib_compiler.get_default_include_dirs()]
    inc_paths = [x.parent for x in inc_paths if x.exists()]
    inc_paths = [x.resolve() for x in inc_paths]
    roots += inc_paths
    if self.env.machines[self.for_machine].is_windows():
        c_root = Path('C:/Boost')
        if c_root.is_dir():
            roots += [c_root]
        prog_files = Path('C:/Program Files/boost')
        local_boost = Path('C:/local')
        candidates: T.List[Path] = []
        if prog_files.is_dir():
            candidates += [*prog_files.iterdir()]
        if local_boost.is_dir():
            candidates += [*local_boost.iterdir()]
        roots += [x for x in candidates if x.name.lower().startswith('boost') and x.is_dir()]
    else:
        tmp: T.List[Path] = []
        tmp += [Path('/opt/local')]
        tmp += [Path('/usr/local/opt/boost')]
        tmp += [Path('/usr/local')]
        tmp += [Path('/usr')]
        tmp = [x for x in tmp if x.is_dir()]
        tmp = [x.resolve() for x in tmp]
        roots += tmp
    self.check_and_set_roots(roots, use_system=True)