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
def extended_iglob(pattern):
    if '{' in pattern:
        m = re.match('(.*){([^}]+)}(.*)', pattern)
        if m:
            before, switch, after = m.groups()
            for case in switch.split(','):
                for path in extended_iglob(before + case + after):
                    yield path
            return
    if '**/' in pattern or (os.sep == '\\' and '**\\' in pattern):
        seen = set()
        first, rest = re.split('\\*\\*[%s]' % ('/\\\\' if os.sep == '\\' else '/'), pattern, 1)
        if first:
            first = iglob(first + os.sep)
        else:
            first = ['']
        for root in first:
            for path in extended_iglob(join_path(root, rest)):
                if path not in seen:
                    seen.add(path)
                    yield path
            for path in extended_iglob(join_path(root, '*', '**', rest)):
                if path not in seen:
                    seen.add(path)
                    yield path
    else:
        for path in iglob(pattern):
            yield path