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
def name_for_module(filename: str, frame: FrameType | None) -> str:
    """Get the name of the module for a filename and frame.

    For configurability's sake, we allow __main__ modules to be matched by
    their importable name.

    If loaded via runpy (aka -m), we can usually recover the "original"
    full dotted module name, otherwise, we resort to interpreting the
    file name to get the module's name.  In the case that the module name
    can't be determined, None is returned.

    """
    module_globals = frame.f_globals if frame is not None else {}
    dunder_name: str = module_globals.get('__name__', None)
    if isinstance(dunder_name, str) and dunder_name != '__main__':
        return dunder_name
    spec = module_globals.get('__spec__', None)
    if spec:
        fullname = spec.name
        if isinstance(fullname, str) and fullname != '__main__':
            return fullname
    inspectedname = inspect.getmodulename(filename)
    if inspectedname is not None:
        return inspectedname
    else:
        return dunder_name