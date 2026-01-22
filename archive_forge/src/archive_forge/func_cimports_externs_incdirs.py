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
@cached_method
def cimports_externs_incdirs(self, filename):
    cimports, includes, externs = self.parse_dependencies(filename)[:3]
    cimports = set(cimports)
    externs = set(externs)
    incdirs = set()
    for include in self.included_files(filename):
        included_cimports, included_externs, included_incdirs = self.cimports_externs_incdirs(include)
        cimports.update(included_cimports)
        externs.update(included_externs)
        incdirs.update(included_incdirs)
    externs, incdir = normalize_existing(filename, externs)
    if incdir:
        incdirs.add(incdir)
    return (tuple(cimports), externs, incdirs)