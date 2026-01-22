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
def create_stub_module(inference_state, grammar, python_value_set, stub_module_node, file_io, import_names):
    if import_names == ('typing',):
        module_cls = TypingModuleWrapper
    else:
        module_cls = StubModuleValue
    file_name = os.path.basename(file_io.path)
    stub_module_value = module_cls(python_value_set, inference_state, stub_module_node, file_io=file_io, string_names=import_names, code_lines=get_cached_code_lines(grammar, file_io.path), is_package=file_name == '__init__.pyi')
    return stub_module_value