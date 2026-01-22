import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def docmodule(self, object, name=None, mod=None):
    """Produce text documentation for a given module object."""
    name = object.__name__
    synop, desc = splitdoc(getdoc(object))
    result = self.section('NAME', name + (synop and ' - ' + synop))
    all = getattr(object, '__all__', None)
    docloc = self.getdocloc(object)
    if docloc is not None:
        result = result + self.section('MODULE REFERENCE', docloc + '\n\nThe following documentation is automatically generated from the Python\nsource files.  It may be incomplete, incorrect or include features that\nare considered implementation detail and may vary between Python\nimplementations.  When in doubt, consult the module reference at the\nlocation listed above.\n')
    if desc:
        result = result + self.section('DESCRIPTION', desc)
    classes = []
    for key, value in inspect.getmembers(object, inspect.isclass):
        if all is not None or (inspect.getmodule(value) or object) is object:
            if visiblename(key, all, object):
                classes.append((key, value))
    funcs = []
    for key, value in inspect.getmembers(object, inspect.isroutine):
        if all is not None or inspect.isbuiltin(value) or inspect.getmodule(value) is object:
            if visiblename(key, all, object):
                funcs.append((key, value))
    data = []
    for key, value in inspect.getmembers(object, isdata):
        if visiblename(key, all, object):
            data.append((key, value))
    modpkgs = []
    modpkgs_names = set()
    if hasattr(object, '__path__'):
        for importer, modname, ispkg in pkgutil.iter_modules(object.__path__):
            modpkgs_names.add(modname)
            if ispkg:
                modpkgs.append(modname + ' (package)')
            else:
                modpkgs.append(modname)
        modpkgs.sort()
        result = result + self.section('PACKAGE CONTENTS', '\n'.join(modpkgs))
    submodules = []
    for key, value in inspect.getmembers(object, inspect.ismodule):
        if value.__name__.startswith(name + '.') and key not in modpkgs_names:
            submodules.append(key)
    if submodules:
        submodules.sort()
        result = result + self.section('SUBMODULES', '\n'.join(submodules))
    if classes:
        classlist = [value for key, value in classes]
        contents = [self.formattree(inspect.getclasstree(classlist, 1), name)]
        for key, value in classes:
            contents.append(self.document(value, key, name))
        result = result + self.section('CLASSES', '\n'.join(contents))
    if funcs:
        contents = []
        for key, value in funcs:
            contents.append(self.document(value, key, name))
        result = result + self.section('FUNCTIONS', '\n'.join(contents))
    if data:
        contents = []
        for key, value in data:
            contents.append(self.docother(value, key, name, maxlen=70))
        result = result + self.section('DATA', '\n'.join(contents))
    if hasattr(object, '__version__'):
        version = str(object.__version__)
        if version[:11] == '$' + 'Revision: ' and version[-1:] == '$':
            version = version[11:-1].strip()
        result = result + self.section('VERSION', version)
    if hasattr(object, '__date__'):
        result = result + self.section('DATE', str(object.__date__))
    if hasattr(object, '__author__'):
        result = result + self.section('AUTHOR', str(object.__author__))
    if hasattr(object, '__credits__'):
        result = result + self.section('CREDITS', str(object.__credits__))
    try:
        file = inspect.getabsfile(object)
    except TypeError:
        file = '(built-in)'
    result = result + self.section('FILE', file)
    return result