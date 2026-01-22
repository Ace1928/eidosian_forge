from typing import Any, Collection, Dict, Optional, Tuple, cast
from ..language import ast, DirectiveLocation
from ..pyutils import inspect, is_description
from .assert_name import assert_name
from .definition import GraphQLArgument, GraphQLInputType, GraphQLNonNull, is_input_type
from .scalars import GraphQLBoolean, GraphQLString
def is_specified_directive(directive: GraphQLDirective) -> bool:
    """Check whether the given directive is one of the specified directives."""
    return any((specified_directive.name == directive.name for specified_directive in specified_directives))