from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_schema_def(definition):
    if not definition:
        raise Exception('def must be defined.')
    handler = _schema_def_handlers.get(type(definition))
    if not handler:
        raise Exception('Type kind "{}" not supported.'.format(type(definition).__name__))
    return handler(definition)