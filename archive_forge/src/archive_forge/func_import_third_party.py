from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
def import_third_party(modname: str) -> tuple[ModuleType, bool]:
    """Import a third-party module we need, but might not be installed.

    This also cleans out the module after the import, so that coverage won't
    appear to have imported it.  This lets the third party use coverage for
    their own tests.

    Arguments:
        modname (str): the name of the module to import.

    Returns:
        The imported module, and a boolean indicating if the module could be imported.

    If the boolean is False, the module returned is not the one you want: don't use it.

    """
    with sys_modules_saved():
        try:
            return (importlib.import_module(modname), True)
        except ImportError:
            return (sys, False)