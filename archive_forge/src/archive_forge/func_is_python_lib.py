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
def is_python_lib(self) -> bool:
    return any((self.mod_name.startswith(x) for x in BoostLibraryFile.boost_python_libs))