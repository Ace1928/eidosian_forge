import glob
import inspect
import os
import re
import sys
from importlib import import_module
from importlib.machinery import all_suffixes
from time import time
from zipimport import zipimporter
from .completer import expand_user, compress_user
from .error import TryNext
from ..utils._process_common import arg_split
from IPython import get_ipython
from typing import List
import_re = re.compile(r'(?P<name>[^\W\d]\w*?)'
def magic_run_completer(self, event):
    """Complete files that end in .py or .ipy or .ipynb for the %run command.
    """
    comps = arg_split(event.line, strict=False)
    if len(comps) > 1 and (not event.line.endswith(' ')):
        relpath = comps[-1].strip('\'"')
    else:
        relpath = ''
    lglob = glob.glob
    isdir = os.path.isdir
    relpath, tilde_expand, tilde_val = expand_user(relpath)
    if any((magic_run_re.match(c) for c in comps)):
        matches = [f.replace('\\', '/') + ('/' if isdir(f) else '') for f in lglob(relpath + '*')]
    else:
        dirs = [f.replace('\\', '/') + '/' for f in lglob(relpath + '*') if isdir(f)]
        pys = [f.replace('\\', '/') for f in lglob(relpath + '*.py') + lglob(relpath + '*.ipy') + lglob(relpath + '*.ipynb') + lglob(relpath + '*.pyw')]
        matches = dirs + pys
    return [compress_user(p, tilde_expand, tilde_val) for p in matches]