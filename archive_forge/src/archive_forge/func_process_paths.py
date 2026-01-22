from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import is_windows, MesonException, PerMachine, stringlistify, extract_as_list
from ..cmake import CMakeExecutor, CMakeTraceParser, CMakeException, CMakeToolchain, CMakeExecScope, check_cmake_args, resolve_cmake_trace_targets, cmake_is_debug
from .. import mlog
import importlib.resources
from pathlib import Path
import functools
import re
import os
import shutil
import textwrap
import typing as T
def process_paths(l: T.List[str]) -> T.Set[str]:
    if is_windows():
        tmp = [x.split(os.pathsep) for x in l]
    else:
        tmp = [re.split(':|;', x) for x in l]
    flattened = [x for sublist in tmp for x in sublist]
    return set(flattened)