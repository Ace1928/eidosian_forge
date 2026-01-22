from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_field_def_map(definition):
    return OrderedDict(((f.name.value, GraphQLField(type=produce_type_def(f.type), args=make_input_values(f.arguments, GraphQLArgument), deprecation_reason=get_deprecation_reason(f.directives))) for f in definition.fields))