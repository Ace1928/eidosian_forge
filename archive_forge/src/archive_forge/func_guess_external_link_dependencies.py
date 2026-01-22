from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def guess_external_link_dependencies(self, linker, target, commands, internal):
    search_dirs = OrderedSet()
    libs = OrderedSet()
    absolute_libs = []
    build_dir = self.environment.get_build_dir()
    it = iter(linker.native_args_to_unix(commands))
    for item in it:
        if item in internal and (not item.startswith('-')):
            continue
        if item.startswith('-L'):
            if len(item) > 2:
                path = item[2:]
            else:
                try:
                    path = next(it)
                except StopIteration:
                    mlog.warning('Generated linker command has -L argument without following path')
                    break
            if not os.path.isabs(path):
                path = os.path.join(build_dir, path)
            search_dirs.add(path)
        elif item.startswith('-l'):
            if len(item) > 2:
                lib = item[2:]
            else:
                try:
                    lib = next(it)
                except StopIteration:
                    mlog.warning("Generated linker command has '-l' argument without following library name")
                    break
            libs.add(lib)
        elif os.path.isabs(item) and self.environment.is_library(item) and os.path.isfile(item):
            absolute_libs.append(item)
    guessed_dependencies = []
    try:
        static_patterns = linker.get_library_naming(self.environment, LibType.STATIC, strict=True)
        shared_patterns = linker.get_library_naming(self.environment, LibType.SHARED, strict=True)
        search_dirs = tuple(search_dirs) + tuple(linker.get_library_dirs(self.environment))
        for libname in libs:
            staticlibs = self.guess_library_absolute_path(linker, libname, search_dirs, static_patterns)
            sharedlibs = self.guess_library_absolute_path(linker, libname, search_dirs, shared_patterns)
            if staticlibs:
                guessed_dependencies.append(staticlibs.resolve().as_posix())
            if sharedlibs:
                guessed_dependencies.append(sharedlibs.resolve().as_posix())
    except (mesonlib.MesonException, AttributeError) as e:
        if 'get_library_naming' not in str(e):
            raise
    return guessed_dependencies + absolute_libs