from asyncio import ensure_future, gather
from collections.abc import Mapping
from inspect import isawaitable
from typing import (
from ..error import GraphQLError, GraphQLFormattedError, located_error
from ..language import (
from ..pyutils import (
from ..type import (
from .collect_fields import collect_fields, collect_sub_fields
from .middleware import MiddlewareManager
from .values import get_argument_values, get_variable_values
def default_type_resolver(value: Any, info: GraphQLResolveInfo, abstract_type: GraphQLAbstractType) -> AwaitableOrValue[Optional[str]]:
    """Default type resolver function.

    If a resolve_type function is not given, then a default resolve behavior is used
    which attempts two strategies:

    First, See if the provided value has a ``__typename`` field defined, if so, use that
    value as name of the resolved type.

    Otherwise, test each possible type for the abstract type by calling
    :meth:`~graphql.type.GraphQLObjectType.is_type_of` for the object
    being coerced, returning the first type that matches.
    """
    type_name = get_typename(value)
    if isinstance(type_name, str):
        return type_name
    possible_types = info.schema.get_possible_types(abstract_type)
    is_awaitable = info.is_awaitable
    awaitable_is_type_of_results: List[Awaitable] = []
    append_awaitable_results = awaitable_is_type_of_results.append
    awaitable_types: List[GraphQLObjectType] = []
    append_awaitable_types = awaitable_types.append
    for type_ in possible_types:
        if type_.is_type_of:
            is_type_of_result = type_.is_type_of(value, info)
            if is_awaitable(is_type_of_result):
                append_awaitable_results(cast(Awaitable, is_type_of_result))
                append_awaitable_types(type_)
            elif is_type_of_result:
                return type_.name
    if awaitable_is_type_of_results:

        async def get_type() -> Optional[str]:
            is_type_of_results = await gather(*awaitable_is_type_of_results)
            for is_type_of_result, type_ in zip(is_type_of_results, awaitable_types):
                if is_type_of_result:
                    return type_.name
            return None
        return get_type()
    return None