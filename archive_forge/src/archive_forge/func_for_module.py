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
@classmethod
def for_module(cls, modname: str) -> 'ModuleAnalyzer':
    if ('module', modname) in cls.cache:
        entry = cls.cache['module', modname]
        if isinstance(entry, PycodeError):
            raise entry
        return entry
    try:
        filename, source = cls.get_module_source(modname)
        if source is not None:
            obj = cls.for_string(source, modname, filename or '<string>')
        elif filename is not None:
            obj = cls.for_file(filename, modname)
    except PycodeError as err:
        cls.cache['module', modname] = err
        raise
    cls.cache['module', modname] = obj
    return obj