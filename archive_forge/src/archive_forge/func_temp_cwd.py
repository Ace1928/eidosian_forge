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
@contextlib.contextmanager
def temp_cwd(name='tempcwd', quiet=False, path=None):
    """
    Context manager that temporarily changes the CWD.

    An existing path may be provided as *path*, in which case this
    function makes no changes to the file system.

    Otherwise, the new CWD is created in the current directory and it's
    named *name*. If *quiet* is False (default) and it's not possible to
    create or change the CWD, an error is raised.  If it's True, only a
    warning is raised and the original CWD is used.
    """
    saved_dir = os.getcwd()
    is_temporary = False
    if path is None:
        path = name
        try:
            os.mkdir(name)
            is_temporary = True
        except OSError:
            if not quiet:
                raise
            warnings.warn('tests may fail, unable to create temp CWD ' + name, RuntimeWarning, stacklevel=3)
    try:
        os.chdir(path)
    except OSError:
        if not quiet:
            raise
        warnings.warn('tests may fail, unable to change the CWD to ' + path, RuntimeWarning, stacklevel=3)
    try:
        yield os.getcwd()
    finally:
        os.chdir(saved_dir)
        if is_temporary:
            rmtree(name)