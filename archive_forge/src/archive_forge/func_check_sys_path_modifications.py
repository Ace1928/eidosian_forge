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
@inference_state_method_cache(default=[])
def check_sys_path_modifications(module_context):
    """
    Detect sys.path modifications within module.
    """

    def get_sys_path_powers(names):
        for name in names:
            power = name.parent.parent
            if power is not None and power.type in ('power', 'atom_expr'):
                c = power.children
                if c[0].type == 'name' and c[0].value == 'sys' and (c[1].type == 'trailer'):
                    n = c[1].children[1]
                    if n.type == 'name' and n.value == 'path':
                        yield (name, power)
    if module_context.tree_node is None:
        return []
    added = []
    try:
        possible_names = module_context.tree_node.get_used_names()['path']
    except KeyError:
        pass
    else:
        for name, power in get_sys_path_powers(possible_names):
            expr_stmt = power.parent
            if len(power.children) >= 4:
                added.extend(_paths_from_list_modifications(module_context, *power.children[2:4]))
            elif expr_stmt is not None and expr_stmt.type == 'expr_stmt':
                added.extend(_paths_from_assignment(module_context, expr_stmt))
    return added