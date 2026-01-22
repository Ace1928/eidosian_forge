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
def pyimport(self, modname=None, ensuresyspath=True):
    """Return path as an imported python module.

        If modname is None, look for the containing package
        and construct an according module name.
        The module will be put/looked up in sys.modules.
        if ensuresyspath is True then the root dir for importing
        the file (taking __init__.py files into account) will
        be prepended to sys.path if it isn't there already.
        If ensuresyspath=="append" the root dir will be appended
        if it isn't already contained in sys.path.
        if ensuresyspath is False no modification of syspath happens.

        Special value of ensuresyspath=="importlib" is intended
        purely for using in pytest, it is capable only of importing
        separate .py files outside packages, e.g. for test suite
        without any __init__.py file. It effectively allows having
        same-named test modules in different places and offers
        mild opt-in via this option. Note that it works only in
        recent versions of python.
        """
    if not self.check():
        raise error.ENOENT(self)
    if ensuresyspath == 'importlib':
        if modname is None:
            modname = self.purebasename
        spec = importlib.util.spec_from_file_location(modname, str(self))
        if spec is None or spec.loader is None:
            raise ImportError(f"Can't find module {modname} at location {self!s}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    pkgpath = None
    if modname is None:
        pkgpath = self.pypkgpath()
        if pkgpath is not None:
            pkgroot = pkgpath.dirpath()
            names = self.new(ext='').relto(pkgroot).split(self.sep)
            if names[-1] == '__init__':
                names.pop()
            modname = '.'.join(names)
        else:
            pkgroot = self.dirpath()
            modname = self.purebasename
        self._ensuresyspath(ensuresyspath, pkgroot)
        __import__(modname)
        mod = sys.modules[modname]
        if self.basename == '__init__.py':
            return mod
        modfile = mod.__file__
        assert modfile is not None
        if modfile[-4:] in ('.pyc', '.pyo'):
            modfile = modfile[:-1]
        elif modfile.endswith('$py.class'):
            modfile = modfile[:-9] + '.py'
        if modfile.endswith(os.sep + '__init__.py'):
            if self.basename != '__init__.py':
                modfile = modfile[:-12]
        try:
            issame = self.samefile(modfile)
        except error.ENOENT:
            issame = False
        if not issame:
            ignore = os.getenv('PY_IGNORE_IMPORTMISMATCH')
            if ignore != '1':
                raise self.ImportMismatchError(modname, modfile, self)
        return mod
    else:
        try:
            return sys.modules[modname]
        except KeyError:
            import types
            mod = types.ModuleType(modname)
            mod.__file__ = str(self)
            sys.modules[modname] = mod
            try:
                with open(str(self), 'rb') as f:
                    exec(f.read(), mod.__dict__)
            except BaseException:
                del sys.modules[modname]
                raise
            return mod