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
def infer_annotation(context, annotation):
    """
    Inferes an annotation node. This means that it inferes the part of
    `int` here:

        foo: int = 3

    Also checks for forward references (strings)
    """
    value_set = context.infer_node(annotation)
    if len(value_set) != 1:
        debug.warning('Inferred typing index %s should lead to 1 object,  not %s' % (annotation, value_set))
        return value_set
    inferred_value = list(value_set)[0]
    if is_string(inferred_value):
        result = _get_forward_reference_node(context, inferred_value.get_safe_value())
        if result is not None:
            return context.infer_node(result)
    return value_set