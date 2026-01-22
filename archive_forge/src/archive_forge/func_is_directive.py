from typing import Any, Collection, Dict, Optional, Tuple, cast
from ..language import ast, DirectiveLocation
from ..pyutils import inspect, is_description
from .assert_name import assert_name
from .definition import GraphQLArgument, GraphQLInputType, GraphQLNonNull, is_input_type
from .scalars import GraphQLBoolean, GraphQLString
def is_directive(directive: Any) -> bool:
    """Test if the given value is a GraphQL directive."""
    return isinstance(directive, GraphQLDirective)