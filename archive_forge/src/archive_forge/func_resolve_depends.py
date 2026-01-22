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
def resolve_depends(depends, include_dirs):
    include_dirs = tuple(include_dirs)
    resolved = []
    for depend in depends:
        path = resolve_depend(depend, include_dirs)
        if path is not None:
            resolved.append(path)
    return resolved