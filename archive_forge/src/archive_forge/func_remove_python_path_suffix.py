import os
import re
from pathlib import Path
from importlib.machinery import all_suffixes
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ContextualizedNode
from jedi.inference.helpers import is_string, get_str_or_none
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FileIO
from jedi import settings
from jedi import debug
def remove_python_path_suffix(path):
    for suffix in all_suffixes() + ['.pyi']:
        if path.suffix == suffix:
            path = path.with_name(path.stem)
            break
    return path