from asyncio import ensure_future
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Dict, Optional, Union, Type, cast
from .error import GraphQLError
from .execution import execute, ExecutionResult, ExecutionContext, Middleware
from .language import parse, Source
from .pyutils import AwaitableOrValue
from .type import (
def graphql_sync(schema: GraphQLSchema, source: Union[str, Source], root_value: Any=None, context_value: Any=None, variable_values: Optional[Dict[str, Any]]=None, operation_name: Optional[str]=None, field_resolver: Optional[GraphQLFieldResolver]=None, type_resolver: Optional[GraphQLTypeResolver]=None, middleware: Optional[Middleware]=None, execution_context_class: Optional[Type[ExecutionContext]]=None, check_sync: bool=False) -> ExecutionResult:
    """Execute a GraphQL operation synchronously.

    The graphql_sync function also fulfills GraphQL operations by parsing, validating,
    and executing a GraphQL document along side a GraphQL schema. However, it guarantees
    to complete synchronously (or throw an error) assuming that all field resolvers
    are also synchronous.

    Set check_sync to True to still run checks that no awaitable values are returned.
    """
    is_awaitable = check_sync if callable(check_sync) else None if check_sync else assume_not_awaitable
    result = graphql_impl(schema, source, root_value, context_value, variable_values, operation_name, field_resolver, type_resolver, middleware, execution_context_class, is_awaitable)
    if isawaitable(result):
        ensure_future(cast(Awaitable[ExecutionResult], result)).cancel()
        raise RuntimeError('GraphQL execution failed to complete synchronously.')
    return cast(ExecutionResult, result)