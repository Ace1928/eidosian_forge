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
def transitive_merge_helper(self, node, extract, merge, seen, stack, outgoing):
    if node in seen:
        return (seen[node], None)
    deps = extract(node)
    if node in stack:
        return (deps, node)
    try:
        stack[node] = len(stack)
        loop = None
        for next in outgoing(node):
            sub_deps, sub_loop = self.transitive_merge_helper(next, extract, merge, seen, stack, outgoing)
            if sub_loop is not None:
                if loop is not None and stack[loop] < stack[sub_loop]:
                    pass
                else:
                    loop = sub_loop
            deps = merge(deps, sub_deps)
        if loop == node:
            loop = None
        if loop is None:
            seen[node] = deps
        return (deps, loop)
    finally:
        del stack[node]