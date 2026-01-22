from the deprecated imp module.
import os
import importlib.util
import importlib.machinery
from importlib.util import module_from_spec
def get_frozen_object(module, paths=None):
    spec = find_spec(module, paths)
    if not spec:
        raise ImportError("Can't find %s" % module)
    return spec.loader.get_code(module)