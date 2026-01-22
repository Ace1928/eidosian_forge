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
def detect_lib_dirs(self, root: Path, use_system: bool) -> T.List[Path]:
    if use_system:
        system_dirs_t = self.clib_compiler.get_library_dirs(self.env)
        system_dirs = [Path(x) for x in system_dirs_t]
        system_dirs = [x.resolve() for x in system_dirs if x.exists()]
        system_dirs = [x for x in system_dirs if mesonlib.path_is_in_root(x, root)]
        system_dirs = list(mesonlib.OrderedSet(system_dirs))
        if system_dirs:
            return system_dirs
    dirs: T.List[Path] = []
    subdirs: T.List[Path] = []
    for i in root.iterdir():
        if i.is_dir() and i.name.startswith('lib'):
            dirs += [i]
    for i in dirs:
        for j in i.iterdir():
            if j.is_dir() and j.name.endswith('-linux-gnu'):
                subdirs += [j]
    if not self.arch:
        return dirs + subdirs
    arch_list_32 = ['32', 'i386']
    arch_list_64 = ['64']
    raw_list = dirs + subdirs
    no_arch = [x for x in raw_list if not any((y in x.name for y in arch_list_32 + arch_list_64))]
    matching_arch: T.List[Path] = []
    if '32' in self.arch:
        matching_arch = [x for x in raw_list if any((y in x.name for y in arch_list_32))]
    elif '64' in self.arch:
        matching_arch = [x for x in raw_list if any((y in x.name for y in arch_list_64))]
    return sorted(matching_arch) + sorted(no_arch)