from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def get_deprecation_reason(directives):
    deprecated_ast = next((directive for directive in directives if directive.name.value == GraphQLDeprecatedDirective.name), None)
    if deprecated_ast:
        args = get_argument_values(GraphQLDeprecatedDirective.args, deprecated_ast.arguments)
        return args['reason']
    else:
        return None