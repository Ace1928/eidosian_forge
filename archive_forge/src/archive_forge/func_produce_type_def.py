from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def produce_type_def(type_ast):
    type_name = _get_named_type_ast(type_ast).name.value
    type_def = type_def_named(type_name)
    return _build_wrapped_type(type_def, type_ast)