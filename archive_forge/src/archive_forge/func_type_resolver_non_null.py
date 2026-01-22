import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def type_resolver_non_null(return_type, resolver, fragment, exe_context, info):
    resolver = type_resolver(return_type.of_type, resolver, fragment, exe_context, info)
    nonnull_complete = partial(complete_nonnull_value, exe_context, info)
    on_resolve_error = partial(on_error, exe_context, info, False)
    return partial(on_complete_resolver, on_resolve_error, nonnull_complete, exe_context, info, resolver)