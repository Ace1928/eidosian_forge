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
def detect_cpu(compilers: CompilersDict) -> str:
    if mesonlib.is_windows():
        trial = detect_windows_arch(compilers)
    elif mesonlib.is_freebsd() or mesonlib.is_netbsd() or mesonlib.is_openbsd() or mesonlib.is_aix():
        trial = platform.processor().lower()
    else:
        trial = platform.machine().lower()
    if trial in {'amd64', 'x64', 'i86pc'}:
        trial = 'x86_64'
    if trial == 'x86_64':
        if any_compiler_has_define(compilers, '__i386__'):
            trial = 'i686'
    elif trial.startswith('aarch64') or trial.startswith('arm64'):
        if any_compiler_has_define(compilers, '__arm__'):
            trial = 'arm'
        else:
            trial = 'aarch64'
    elif trial.startswith('earm'):
        trial = 'arm'
    elif trial == 'e2k':
        trial = platform.processor().lower()
    elif trial.startswith('mips'):
        if '64' not in trial:
            trial = 'mips'
        elif compilers and (not any_compiler_has_define(compilers, '__mips64')):
            trial = 'mips'
        else:
            trial = 'mips64'
    elif trial == 'ppc':
        if any_compiler_has_define(compilers, '__64BIT__'):
            trial = 'ppc64'
    return trial