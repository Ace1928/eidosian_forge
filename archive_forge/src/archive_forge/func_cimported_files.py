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
def cimported_files(self, filename):
    filename_root, filename_ext = os.path.splitext(filename)
    if filename_ext in ('.pyx', '.py') and path_exists(filename_root + '.pxd'):
        pxd_list = [filename_root + '.pxd']
    else:
        pxd_list = []
    for module in self.cimports(filename):
        if module[:7] == 'cython.' or module == 'cython':
            continue
        pxd_file = self.find_pxd(module, filename)
        if pxd_file is not None:
            pxd_list.append(pxd_file)
    return tuple(pxd_list)