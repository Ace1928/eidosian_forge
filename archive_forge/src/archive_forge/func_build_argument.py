from itertools import chain
from typing import cast, Callable, Collection, Dict, List, Union
from ..language import DirectiveLocation, parse_value
from ..pyutils import inspect, Undefined
from ..type import (
from .get_introspection_query import (
from .value_from_ast import value_from_ast
def build_argument(argument_introspection: IntrospectionInputValue) -> GraphQLArgument:
    type_introspection = cast(IntrospectionType, argument_introspection['type'])
    type_ = get_type(type_introspection)
    if not is_input_type(type_):
        raise TypeError(f'Introspection must provide input type for arguments, but received: {inspect(type_)}.')
    type_ = cast(GraphQLInputType, type_)
    default_value_introspection = argument_introspection.get('defaultValue')
    default_value = Undefined if default_value_introspection is None else value_from_ast(parse_value(default_value_introspection), type_)
    return GraphQLArgument(type_, default_value=default_value, description=argument_introspection.get('description'), deprecation_reason=argument_introspection.get('deprecationReason'))