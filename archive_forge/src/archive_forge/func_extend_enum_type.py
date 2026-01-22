from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def extend_enum_type(type_: GraphQLEnumType) -> GraphQLEnumType:
    kwargs = type_.to_kwargs()
    extensions = tuple(type_extensions_map[kwargs['name']])
    return GraphQLEnumType(**merge_kwargs(kwargs, values={**kwargs['values'], **build_enum_value_map(extensions)}, extension_ast_nodes=kwargs['extension_ast_nodes'] + extensions))