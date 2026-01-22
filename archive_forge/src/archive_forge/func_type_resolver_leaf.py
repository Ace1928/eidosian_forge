import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def type_resolver_leaf(return_type, resolver, exe_context, info, catch_error):
    leaf_complete = partial(complete_leaf_value, return_type.serialize)
    on_resolve_error = partial(on_error, exe_context, info, catch_error)
    return partial(on_complete_resolver, on_resolve_error, leaf_complete, exe_context, info, resolver)