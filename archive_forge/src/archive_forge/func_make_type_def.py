from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_type_def(definition):
    return GraphQLObjectType(name=definition.name.value, fields=lambda: make_field_def_map(definition), interfaces=make_implemented_interfaces(definition))