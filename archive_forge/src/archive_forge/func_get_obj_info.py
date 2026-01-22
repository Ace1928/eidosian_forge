import importlib
from abc import ABC, abstractmethod
from pickle import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
from ._mangling import demangle, get_mangle_prefix, is_mangled
def get_obj_info(obj):
    assert name is not None
    module_name = self.whichmodule(obj, name)
    is_mangled_ = is_mangled(module_name)
    location = get_mangle_prefix(module_name) if is_mangled_ else 'the current Python environment'
    importer_name = f'the importer for {get_mangle_prefix(module_name)}' if is_mangled_ else "'sys_importer'"
    return (module_name, location, importer_name)