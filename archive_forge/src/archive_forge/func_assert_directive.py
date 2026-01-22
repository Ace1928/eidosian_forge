from typing import Any, Collection, Dict, Optional, Tuple, cast
from ..language import ast, DirectiveLocation
from ..pyutils import inspect, is_description
from .assert_name import assert_name
from .definition import GraphQLArgument, GraphQLInputType, GraphQLNonNull, is_input_type
from .scalars import GraphQLBoolean, GraphQLString
def assert_directive(directive: Any) -> GraphQLDirective:
    if not is_directive(directive):
        raise TypeError(f'Expected {inspect(directive)} to be a GraphQL directive.')
    return cast(GraphQLDirective, directive)