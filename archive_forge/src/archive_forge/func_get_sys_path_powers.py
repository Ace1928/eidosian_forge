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
def get_sys_path_powers(names):
    for name in names:
        power = name.parent.parent
        if power is not None and power.type in ('power', 'atom_expr'):
            c = power.children
            if c[0].type == 'name' and c[0].value == 'sys' and (c[1].type == 'trailer'):
                n = c[1].children[1]
                if n.type == 'name' and n.value == 'path':
                    yield (name, power)