from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_default_value(f):
    default_value = f.get('defaultValue')
    if default_value is None:
        return None
    return value_from_ast(parse_value(default_value), get_input_type(f['type']))