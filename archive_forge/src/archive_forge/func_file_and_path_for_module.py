from __future__ import annotations
import importlib.util
import inspect
import itertools
import os
import platform
import re
import sys
import sysconfig
import traceback
from types import FrameType, ModuleType
from typing import (
from coverage import env
from coverage.disposition import FileDisposition, disposition_init
from coverage.exceptions import CoverageException, PluginError
from coverage.files import TreeMatcher, GlobMatcher, ModuleMatcher
from coverage.files import prep_patterns, find_python_files, canonical_filename
from coverage.misc import sys_modules_saved
from coverage.python import source_for_file, source_for_morf
from coverage.types import TFileDisposition, TMorf, TWarnFn, TDebugCtl
def file_and_path_for_module(modulename: str) -> tuple[str | None, list[str]]:
    """Find the file and search path for `modulename`.

    Returns:
        filename: The filename of the module, or None.
        path: A list (possibly empty) of directories to find submodules in.

    """
    filename = None
    path = []
    try:
        spec = importlib.util.find_spec(modulename)
    except Exception:
        pass
    else:
        if spec is not None:
            filename = spec.origin
            path = list(spec.submodule_search_locations or ())
    return (filename, path)