import re
from collections.abc import Iterable
from functools import partial
from typing import Type
from graphql_relay import connection_from_array
from ..types import Boolean, Enum, Int, Interface, List, NonNull, Scalar, String, Union
from ..types.field import Field
from ..types.objecttype import ObjectType, ObjectTypeOptions
from ..utils.thenables import maybe_thenable
from .node import is_node, AbstractNode
def get_edge_class(connection_class: Type['Connection'], _node: Type[AbstractNode], base_name: str, strict_types: bool=False):
    edge_class = getattr(connection_class, 'Edge', None)

    class EdgeBase:
        node = Field(NonNull(_node) if strict_types else _node, description='The item at the end of the edge')
        cursor = String(required=True, description='A cursor for use in pagination')

    class EdgeMeta:
        description = f'A Relay edge containing a `{base_name}` and its cursor.'
    edge_name = f'{base_name}Edge'
    edge_bases = [edge_class, EdgeBase] if edge_class else [EdgeBase]
    if not isinstance(edge_class, ObjectType):
        edge_bases = [*edge_bases, ObjectType]
    return type(edge_name, tuple(edge_bases), {'Meta': EdgeMeta})