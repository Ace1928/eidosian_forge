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
def find_pxd(self, module, filename=None):
    is_relative = module[0] == '.'
    if is_relative and (not filename):
        raise NotImplementedError('New relative imports.')
    if filename is not None:
        module_path = module.split('.')
        if is_relative:
            module_path.pop(0)
        package_path = list(self.package(filename))
        while module_path and (not module_path[0]):
            try:
                package_path.pop()
            except IndexError:
                return None
            module_path.pop(0)
        relative = '.'.join(package_path + module_path)
        pxd = self.context.find_pxd_file(relative, source_file_path=filename)
        if pxd:
            return pxd
    if is_relative:
        return None
    return self.context.find_pxd_file(module, source_file_path=filename)