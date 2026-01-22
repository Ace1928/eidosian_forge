from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def import_fresh_module(name, fresh=(), blocked=(), deprecated=False):
    """Import and return a module, deliberately bypassing sys.modules.
    This function imports and returns a fresh copy of the named Python module
    by removing the named module from sys.modules before doing the import.
    Note that unlike reload, the original module is not affected by
    this operation.

    *fresh* is an iterable of additional module names that are also removed
    from the sys.modules cache before doing the import.

    *blocked* is an iterable of module names that are replaced with None
    in the module cache during the import to ensure that attempts to import
    them raise ImportError.

    The named module and any modules named in the *fresh* and *blocked*
    parameters are saved before starting the import and then reinserted into
    sys.modules when the fresh import is complete.

    Module and package deprecation messages are suppressed during this import
    if *deprecated* is True.

    This function will raise ImportError if the named module cannot be
    imported.

    If deprecated is True, any module or package deprecation messages
    will be suppressed.
    """
    with _ignore_deprecated_imports(deprecated):
        orig_modules = {}
        names_to_remove = []
        _save_and_remove_module(name, orig_modules)
        try:
            for fresh_name in fresh:
                _save_and_remove_module(fresh_name, orig_modules)
            for blocked_name in blocked:
                if not _save_and_block_module(blocked_name, orig_modules):
                    names_to_remove.append(blocked_name)
            fresh_module = importlib.import_module(name)
        except ImportError:
            fresh_module = None
        finally:
            for orig_name, module in orig_modules.items():
                sys.modules[orig_name] = module
            for name_to_remove in names_to_remove:
                del sys.modules[name_to_remove]
        return fresh_module