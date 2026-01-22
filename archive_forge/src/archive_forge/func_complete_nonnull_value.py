import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def complete_nonnull_value(exe_context, info, result):
    if result is None:
        raise GraphQLError('Cannot return null for non-nullable field {}.{}.'.format(info.parent_type, info.field_name), info.field_asts)
    return result