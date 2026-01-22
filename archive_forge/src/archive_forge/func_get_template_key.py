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
@staticmethod
def get_template_key(name: str) -> str:
    return 'tmpl_' + sha1(name.encode('utf-8')).hexdigest()