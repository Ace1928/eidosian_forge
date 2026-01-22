import re
import tokenize
from collections import OrderedDict
from importlib import import_module
from inspect import Signature
from os import path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile
from sphinx.errors import PycodeError
from sphinx.pycode.parser import Parser
@staticmethod
def get_module_source(modname: str) -> Tuple[Optional[str], Optional[str]]:
    """Try to find the source code for a module.

        Returns ('filename', 'source'). One of it can be None if
        no filename or source found
        """
    try:
        mod = import_module(modname)
    except Exception as err:
        raise PycodeError('error importing %r' % modname, err) from err
    loader = getattr(mod, '__loader__', None)
    filename = getattr(mod, '__file__', None)
    if loader and getattr(loader, 'get_source', None):
        try:
            source = loader.get_source(modname)
            if source:
                return (filename, source)
        except ImportError:
            pass
    if filename is None and loader and getattr(loader, 'get_filename', None):
        try:
            filename = loader.get_filename(modname)
        except ImportError as err:
            raise PycodeError('error getting filename for %r' % modname, err) from err
    if filename is None:
        raise PycodeError('no source found for module %r' % modname)
    filename = path.normpath(path.abspath(filename))
    if filename.lower().endswith(('.pyo', '.pyc')):
        filename = filename[:-1]
        if not path.isfile(filename) and path.isfile(filename + 'w'):
            filename += 'w'
    elif not filename.lower().endswith(('.py', '.pyw')):
        raise PycodeError('source is not a .py file: %r' % filename)
    elif '.egg' + path.sep in filename:
        pat = '(?<=\\.egg)' + re.escape(path.sep)
        eggpath, _ = re.split(pat, filename, 1)
        if path.isfile(eggpath):
            return (filename, None)
    if not path.isfile(filename):
        raise PycodeError('source file is not present: %r' % filename)
    return (filename, None)