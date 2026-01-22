from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
@cython.locals(start=cython.Py_ssize_t, end=cython.Py_ssize_t)
def line_iter(source):
    if isinstance(source, basestring):
        start = 0
        while True:
            end = source.find('\n', start)
            if end == -1:
                yield source[start:]
                return
            yield source[start:end]
            start = end + 1
    else:
        for line in source:
            yield line