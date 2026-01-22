import re
from textwrap import dedent
from inspect import Parameter
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
def complete_trailer(user_context, values):
    completion_names = []
    for value in values:
        for filter in value.get_filters(origin_scope=user_context.tree_node):
            completion_names += filter.values()
        if not value.is_stub() and isinstance(value, TreeInstance):
            completion_names += _complete_getattr(user_context, value)
    python_values = convert_values(values)
    for c in python_values:
        if c not in values:
            for filter in c.get_filters(origin_scope=user_context.tree_node):
                completion_names += filter.values()
    return completion_names