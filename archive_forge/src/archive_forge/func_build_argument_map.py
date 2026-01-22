from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def build_argument_map(args: Optional[Collection[InputValueDefinitionNode]]) -> GraphQLArgumentMap:
    arg_map: GraphQLArgumentMap = {}
    for arg in args or []:
        type_ = cast(GraphQLInputType, get_wrapped_type(arg.type))
        arg_map[arg.name.value] = GraphQLArgument(type_=type_, description=arg.description.value if arg.description else None, default_value=value_from_ast(arg.default_value, type_), deprecation_reason=get_deprecation_reason(arg), ast_node=arg)
    return arg_map