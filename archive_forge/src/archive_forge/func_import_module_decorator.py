import os
import re
from functools import wraps
from collections import namedtuple
from typing import Dict, Mapping, Tuple
from pathlib import Path
from jedi import settings
from jedi.file_io import FileIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.stub_value import TypingModuleWrapper, StubModuleValue
from jedi.inference.value import ModuleValue
def import_module_decorator(func):

    @wraps(func)
    def wrapper(inference_state, import_names, parent_module_value, sys_path, prefer_stubs):
        python_value_set = inference_state.module_cache.get(import_names)
        if python_value_set is None:
            if parent_module_value is not None and parent_module_value.is_stub():
                parent_module_values = parent_module_value.non_stub_value_set
            else:
                parent_module_values = [parent_module_value]
            if import_names == ('os', 'path'):
                python_value_set = ValueSet.from_sets((func(inference_state, (n,), None, sys_path) for n in ['posixpath', 'ntpath', 'macpath', 'os2emxpath']))
            else:
                python_value_set = ValueSet.from_sets((func(inference_state, import_names, p, sys_path) for p in parent_module_values))
            inference_state.module_cache.add(import_names, python_value_set)
        if not prefer_stubs or import_names[0] in settings.auto_import_modules:
            return python_value_set
        stub = try_to_load_stub_cached(inference_state, import_names, python_value_set, parent_module_value, sys_path)
        if stub is not None:
            return ValueSet([stub])
        return python_value_set
    return wrapper