import copy
import os
import sys
from importlib import import_module
from importlib.util import find_spec as importlib_find
def cached_import(module_path, class_name):
    if not ((module := sys.modules.get(module_path)) and (spec := getattr(module, '__spec__', None)) and (getattr(spec, '_initializing', False) is False)):
        module = import_module(module_path)
    return getattr(module, class_name)