from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
@lru_cache(maxsize=None)
def rpaths_for_non_system_absolute_shared_libraries(self, target: build.BuildTarget, exclude_system: bool=True) -> 'ImmutableListProtocol[str]':
    paths: OrderedSet[str] = OrderedSet()
    srcdir = self.environment.get_source_dir()
    for dep in target.external_deps:
        if dep.type_name not in {'library', 'pkgconfig', 'cmake'}:
            continue
        for libpath in dep.link_args:
            if not os.path.isabs(libpath):
                continue
            libdir = os.path.dirname(libpath)
            if exclude_system and self._libdir_is_system(libdir, target.compilers, self.environment):
                continue
            if libdir in self.get_external_rpath_dirs(target):
                continue
            if not (os.path.splitext(libpath)[1] in {'.dll', '.lib', '.so', '.dylib'} or re.match('.+\\.so(\\.|$)', os.path.basename(libpath))):
                continue
            try:
                commonpath = os.path.commonpath((libdir, srcdir))
            except ValueError:
                commonpath = ''
            if commonpath == srcdir:
                rel_to_src = libdir[len(srcdir) + 1:]
                assert not os.path.isabs(rel_to_src), f'rel_to_src: {rel_to_src} is absolute'
                paths.add(os.path.join(self.build_to_src, rel_to_src))
            else:
                paths.add(libdir)
        paths.difference_update(self.get_rpath_dirs_from_link_args(dep.link_args))
    for i in chain(target.link_targets, target.link_whole_targets):
        if isinstance(i, build.BuildTarget):
            paths.update(self.rpaths_for_non_system_absolute_shared_libraries(i, exclude_system))
    return list(paths)