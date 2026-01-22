from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_union_def(definition):
    return GraphQLUnionType(name=definition.name.value, resolve_type=_none, types=[produce_type_def(t) for t in definition.types])