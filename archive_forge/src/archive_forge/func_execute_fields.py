import collections
from collections.abc import Iterable
import functools
import logging
import sys
from wandb_promise import Promise, promise_for_dict, is_thenable
from ..error import GraphQLError, GraphQLLocatedError
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from .base import (ExecutionContext, ExecutionResult, ResolveInfo, Undefined,
from .executors.sync import SyncExecutor
from .experimental.executor import execute as experimental_execute
from .middleware import MiddlewareManager
def execute_fields(exe_context, parent_type, source_value, fields):
    contains_promise = False
    final_results = OrderedDict()
    for response_name, field_asts in fields.items():
        result = resolve_field(exe_context, parent_type, source_value, field_asts)
        if result is Undefined:
            continue
        final_results[response_name] = result
        if is_thenable(result):
            contains_promise = True
    if not contains_promise:
        return final_results
    return promise_for_dict(final_results)