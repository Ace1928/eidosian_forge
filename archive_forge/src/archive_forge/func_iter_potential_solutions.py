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
def iter_potential_solutions():
    for p in sys_path:
        if str(module_path).startswith(p):
            rest = str(module_path)[len(p):]
            if rest.startswith(os.path.sep) or rest.startswith('/'):
                rest = rest[1:]
            if rest:
                split = rest.split(os.path.sep)
                if not all(split):
                    return
                yield tuple((re.sub('-stubs$', '', s) for s in split))