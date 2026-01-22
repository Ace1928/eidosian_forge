import sys
import os
from distutils.errors import DistutilsPlatformError
from distutils.dep_util import newer, newer_group
from distutils import log
from distutils.command import build_ext as _build_ext
from distutils import sysconfig
import inspect
import warnings

        Walk the list of source files in 'sources', looking for Cython
        source files (.pyx and .py).  Run Cython on all that are
        found, and return a modified 'sources' list with Cython source
        files replaced by the generated C (or C++) files.
        