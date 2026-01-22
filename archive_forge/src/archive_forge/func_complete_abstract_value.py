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
def complete_abstract_value(exe_context, return_type, field_asts, info, result):
    """
    Complete an value of an abstract type by determining the runtime type of that value, then completing based
    on that type.
    """
    runtime_type = None
    if isinstance(return_type, (GraphQLInterfaceType, GraphQLUnionType)):
        if return_type.resolve_type:
            runtime_type = return_type.resolve_type(result, exe_context.context_value, info)
        else:
            runtime_type = get_default_resolve_type_fn(result, exe_context.context_value, info, return_type)
    if isinstance(runtime_type, str):
        runtime_type = info.schema.get_type(runtime_type)
    if not isinstance(runtime_type, GraphQLObjectType):
        raise GraphQLError(('Abstract type {} must resolve to an Object type at runtime ' + 'for field {}.{} with value "{}", received "{}".').format(return_type, info.parent_type, info.field_name, result, runtime_type), field_asts)
    if not exe_context.schema.is_possible_type(return_type, runtime_type):
        raise GraphQLError(u'Runtime Object type "{}" is not a possible type for "{}".'.format(runtime_type, return_type), field_asts)
    return complete_object_value(exe_context, runtime_type, field_asts, info, result)