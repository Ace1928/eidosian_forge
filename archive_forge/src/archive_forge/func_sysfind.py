from __future__ import annotations
import atexit
from contextlib import contextmanager
import fnmatch
import importlib.util
import io
import os
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import isabs
from os.path import isdir
from os.path import isfile
from os.path import islink
from os.path import normpath
import posixpath
from stat import S_ISDIR
from stat import S_ISLNK
from stat import S_ISREG
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import overload
from typing import TYPE_CHECKING
import uuid
import warnings
from . import error
@classmethod
def sysfind(cls, name, checker=None, paths=None):
    """Return a path object found by looking at the systems
        underlying PATH specification. If the checker is not None
        it will be invoked to filter matching paths.  If a binary
        cannot be found, None is returned
        Note: This is probably not working on plain win32 systems
        but may work on cygwin.
        """
    if isabs(name):
        p = local(name)
        if p.check(file=1):
            return p
    else:
        if paths is None:
            if iswin32:
                paths = os.environ['Path'].split(';')
                if '' not in paths and '.' not in paths:
                    paths.append('.')
                try:
                    systemroot = os.environ['SYSTEMROOT']
                except KeyError:
                    pass
                else:
                    paths = [path.replace('%SystemRoot%', systemroot) for path in paths]
            else:
                paths = os.environ['PATH'].split(':')
        tryadd = []
        if iswin32:
            tryadd += os.environ['PATHEXT'].split(os.pathsep)
        tryadd.append('')
        for x in paths:
            for addext in tryadd:
                p = local(x).join(name, abs=True) + addext
                try:
                    if p.check(file=1):
                        if checker:
                            if not checker(p):
                                continue
                        return p
                except error.EACCES:
                    pass
    return None