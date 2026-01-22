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
def detect_lcov(lcov_exe: str='lcov', log: bool=False):
    try:
        p, found = Popen_safe([lcov_exe, '--version'])[0:2]
    except (FileNotFoundError, PermissionError):
        return (None, None)
    found = search_version(found)
    if p.returncode == 0 and found:
        if log:
            mlog.log('Found lcov-{} at {}'.format(found, quote_arg(shutil.which(lcov_exe))))
        return (lcov_exe, found)
    return (None, None)