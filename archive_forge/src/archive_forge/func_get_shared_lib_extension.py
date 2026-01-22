import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def get_shared_lib_extension(is_python_ext=False):
    """Return the correct file extension for shared libraries.

    Parameters
    ----------
    is_python_ext : bool, optional
        Whether the shared library is a Python extension.  Default is False.

    Returns
    -------
    so_ext : str
        The shared library extension.

    Notes
    -----
    For Python shared libs, `so_ext` will typically be '.so' on Linux and OS X,
    and '.pyd' on Windows.  For Python >= 3.2 `so_ext` has a tag prepended on
    POSIX systems according to PEP 3149.

    """
    confvars = distutils.sysconfig.get_config_vars()
    so_ext = confvars.get('EXT_SUFFIX', '')
    if not is_python_ext:
        if sys.platform.startswith('linux') or sys.platform.startswith('gnukfreebsd'):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        elif 'SOABI' in confvars:
            so_ext = so_ext.replace('.' + confvars.get('SOABI'), '', 1)
    return so_ext