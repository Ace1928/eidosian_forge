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
@inference_state_method_cache()
def infer_param(function_value, param, ignore_stars=False):
    values = _infer_param(function_value, param)
    if ignore_stars or not values:
        return values
    inference_state = function_value.inference_state
    if param.star_count == 1:
        tuple_ = builtin_from_name(inference_state, 'tuple')
        return ValueSet([GenericClass(tuple_, TupleGenericManager((values,)))])
    elif param.star_count == 2:
        dct = builtin_from_name(inference_state, 'dict')
        generics = (ValueSet([builtin_from_name(inference_state, 'str')]), values)
        return ValueSet([GenericClass(dct, TupleGenericManager(generics))])
    return values