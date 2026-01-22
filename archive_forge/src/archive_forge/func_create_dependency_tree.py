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
def create_dependency_tree(ctx=None, quiet=False):
    global _dep_tree
    if _dep_tree is None:
        if ctx is None:
            ctx = Context(['.'], get_directive_defaults(), options=CompilationOptions(default_options))
        _dep_tree = DependencyTree(ctx, quiet=quiet)
    return _dep_tree