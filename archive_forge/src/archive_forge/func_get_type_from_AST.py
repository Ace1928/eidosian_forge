from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def get_type_from_AST(astNode):
    type = _get_named_type(astNode.name.value)
    if not type:
        raise GraphQLError(('Unknown type: "{}". Ensure that this type exists ' + 'either in the original schema, or is added in a type definition.').format(astNode.name.value), [astNode])
    return type