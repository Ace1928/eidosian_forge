from ..language.parser import parse_value
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean, GraphQLEnumType,
from ..type.directives import DirectiveLocation, GraphQLDirective
from ..type.introspection import (TypeKind, __Directive, __DirectiveLocation,
from .value_from_ast import value_from_ast
def build_input_object_def(input_object_introspection):
    return GraphQLInputObjectType(name=input_object_introspection['name'], description=input_object_introspection.get('description'), fields=lambda: build_input_value_def_map(input_object_introspection.get('inputFields'), GraphQLInputObjectField))