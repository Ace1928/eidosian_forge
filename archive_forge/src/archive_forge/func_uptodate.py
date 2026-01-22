import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType
from .exceptions import TemplateNotFound
from .utils import internalcode
def uptodate() -> bool:
    try:
        return os.path.getmtime(filename) == mtime
    except OSError:
        return False