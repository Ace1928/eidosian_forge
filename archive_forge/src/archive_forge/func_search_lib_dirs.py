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
def search_lib_dirs(path: str) -> bool:
    for i in [os.path.join(path, x) for x in self.cmakeinfo.common_paths]:
        if not self._cached_isdir(i):
            continue
        cm_dir = os.path.join(i, 'cmake')
        if self._cached_isdir(cm_dir):
            content = self._cached_listdir(cm_dir)
            content = tuple((x for x in content if x[1].startswith(lname)))
            for k in content:
                if find_module(os.path.join(cm_dir, k[0])):
                    return True
        content = self._cached_listdir(i)
        content = tuple((x for x in content if x[1].startswith(lname)))
        for k in content:
            if find_module(os.path.join(i, k[0])):
                return True
    return False