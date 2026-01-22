from jedi import settings
from jedi import debug
from jedi.parser_utils import get_parent_scope
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import TreeArguments
from jedi.inference.param import get_executed_param_names
from jedi.inference.helpers import is_stdlib_path
from jedi.inference.utils import to_list
from jedi.inference.value import instance
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.references import get_module_contexts_containing_name
from jedi.inference import recursion
@debug.increase_indent
@_avoid_recursions
def dynamic_param_lookup(function_value, param_index):
    """
    A dynamic search for param values. If you try to complete a type:

    >>> def func(foo):
    ...     foo
    >>> func(1)
    >>> func("")

    It is not known what the type ``foo`` without analysing the whole code. You
    have to look for all calls to ``func`` to find out what ``foo`` possibly
    is.
    """
    if not function_value.inference_state.do_dynamic_params_search:
        return NO_VALUES
    funcdef = function_value.tree_node
    path = function_value.get_root_context().py__file__()
    if path is not None and is_stdlib_path(path):
        return NO_VALUES
    if funcdef.type == 'lambdef':
        string_name = _get_lambda_name(funcdef)
        if string_name is None:
            return NO_VALUES
    else:
        string_name = funcdef.name.value
    debug.dbg('Dynamic param search in %s.', string_name, color='MAGENTA')
    module_context = function_value.get_root_context()
    arguments_list = _search_function_arguments(module_context, funcdef, string_name)
    values = ValueSet.from_sets((get_executed_param_names(function_value, arguments)[param_index].infer() for arguments in arguments_list))
    debug.dbg('Dynamic param result finished', color='MAGENTA')
    return values