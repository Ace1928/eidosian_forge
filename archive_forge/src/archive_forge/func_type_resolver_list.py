import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def type_resolver_list(return_type, resolver, fragment, exe_context, info, catch_error):
    item_type = return_type.of_type
    inner_resolver = type_resolver(item_type, lambda item: item, fragment, exe_context, info, catch_error=True)
    on_resolve_error = partial(on_error, exe_context, info, catch_error)
    list_complete = partial(complete_list_value, inner_resolver, exe_context, info, on_resolve_error)
    return partial(on_complete_resolver, on_resolve_error, list_complete, exe_context, info, resolver)