import re
from collections import namedtuple
from textwrap import dedent
from itertools import chain
from functools import wraps
from inspect import Parameter
from parso.python.parser import Parser
from parso.python import tree
from jedi.inference.base_value import NO_VALUES
from jedi.inference.syntax_tree import infer_atom
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.compiled import get_string_value_set
from jedi.cache import signature_time_cache, memoize_method
from jedi.parser_utils import get_parent_scope
def filter_follow_imports(names, follow_builtin_imports=False):
    for name in names:
        if name.is_import():
            new_names = list(filter_follow_imports(name.goto(), follow_builtin_imports=follow_builtin_imports))
            found_builtin = False
            if follow_builtin_imports:
                for new_name in new_names:
                    if new_name.start_pos is None:
                        found_builtin = True
            if found_builtin:
                yield name
            else:
                yield from new_names
        else:
            yield name