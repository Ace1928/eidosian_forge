from typing import Any, Callable, NamedTuple, Optional, Union
from graphql_relay.utils.base64 import base64, unbase64
from graphql import (
def to_global_id(type_: str, id_: Union[str, int]) -> str:
    """
    Takes a type name and an ID specific to that type name, and returns a
    "global ID" that is unique among all types.
    """
    return base64(f'{type_}:{GraphQLID.serialize(id_)}')