import sys
from typing import List
from pathlib import Path
from parso.tree import search_ancestor
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.imports import goto_import, load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.helpers import infer_call_of_leaf
def goto_anonymous_param(func):

    def wrapper(param_name):
        is_pytest_param, param_name_is_function_name = _is_a_pytest_param_and_inherited(param_name)
        if is_pytest_param:
            names = _goto_pytest_fixture(param_name.get_root_context(), param_name.string_name, skip_own_module=param_name_is_function_name)
            if names:
                return names
        return func(param_name)
    return wrapper