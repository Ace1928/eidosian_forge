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
@classmethod
def resolve_connection(cls, connection_type, args, resolved):
    if isinstance(resolved, connection_type):
        return resolved
    assert isinstance(resolved, Iterable), f'Resolved value from the connection field has to be an iterable or instance of {connection_type}. Received "{resolved}"'
    connection = connection_from_array(resolved, args, connection_type=partial(connection_adapter, connection_type), edge_type=connection_type.Edge, page_info_type=page_info_adapter)
    connection.iterable = resolved
    return connection