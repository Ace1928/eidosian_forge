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
def try_to_load_stub_cached(inference_state, import_names, *args, **kwargs):
    if import_names is None:
        return None
    try:
        return inference_state.stub_module_cache[import_names]
    except KeyError:
        pass
    inference_state.stub_module_cache[import_names] = None
    inference_state.stub_module_cache[import_names] = result = _try_to_load_stub(inference_state, import_names, *args, **kwargs)
    return result