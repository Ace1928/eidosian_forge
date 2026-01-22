import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def tempfilepager(text, cmd):
    """Page through text by invoking a program on a temporary file."""
    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, 'pydoc.out')
        with open(filename, 'w', errors='backslashreplace', encoding=os.device_encoding(0) if sys.platform == 'win32' else None) as file:
            file.write(text)
        os.system(cmd + ' "' + filename + '"')