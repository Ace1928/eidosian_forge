from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def get_compiler_system_include_dirs(self, for_machine: MachineChoice) -> T.List[str]:
    for comp in self.coredata.compilers[for_machine].values():
        if comp.id == 'clang':
            break
        elif comp.id == 'gcc':
            break
    else:
        return []
    return comp.get_default_include_dirs()