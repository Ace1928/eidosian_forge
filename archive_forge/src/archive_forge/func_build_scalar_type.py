from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def build_scalar_type(type_ast):
    return GraphQLScalarType(type_ast.name.value, serialize=lambda *args, **kwargs: None, parse_value=lambda *args, **kwargs: False, parse_literal=lambda *args, **kwargs: False)