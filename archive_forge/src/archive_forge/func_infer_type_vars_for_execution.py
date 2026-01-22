import re
from inspect import Parameter
from parso import ParserSyntaxError, parse
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.base import DefineGenericBaseClass, GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.gradual.type_var import TypeVar
from jedi.inference.helpers import is_string
from jedi.inference.compiled import builtin_from_name
from jedi.inference.param import get_executed_param_names
from jedi import debug
from jedi import parser_utils
def infer_type_vars_for_execution(function, arguments, annotation_dict):
    """
    Some functions use type vars that are not defined by the class, but rather
    only defined in the function. See for example `iter`. In those cases we
    want to:

    1. Search for undefined type vars.
    2. Infer type vars with the execution state we have.
    3. Return the union of all type vars that have been found.
    """
    context = function.get_default_param_context()
    annotation_variable_results = {}
    executed_param_names = get_executed_param_names(function, arguments)
    for executed_param_name in executed_param_names:
        try:
            annotation_node = annotation_dict[executed_param_name.string_name]
        except KeyError:
            continue
        annotation_variables = find_unknown_type_vars(context, annotation_node)
        if annotation_variables:
            annotation_value_set = context.infer_node(annotation_node)
            kind = executed_param_name.get_kind()
            actual_value_set = executed_param_name.infer()
            if kind is Parameter.VAR_POSITIONAL:
                actual_value_set = actual_value_set.merge_types_of_iterate()
            elif kind is Parameter.VAR_KEYWORD:
                actual_value_set = actual_value_set.try_merge('_dict_values')
            merge_type_var_dicts(annotation_variable_results, annotation_value_set.infer_type_vars(actual_value_set))
    return annotation_variable_results