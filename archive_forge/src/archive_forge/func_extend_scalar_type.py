from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def extend_scalar_type(type_: GraphQLScalarType) -> GraphQLScalarType:
    kwargs = type_.to_kwargs()
    extensions = tuple(type_extensions_map[kwargs['name']])
    specified_by_url = kwargs['specified_by_url']
    for extension_node in extensions:
        specified_by_url = get_specified_by_url(extension_node) or specified_by_url
    return GraphQLScalarType(**merge_kwargs(kwargs, specified_by_url=specified_by_url, extension_ast_nodes=kwargs['extension_ast_nodes'] + extensions))