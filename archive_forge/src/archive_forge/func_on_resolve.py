from wandb_promise import Promise
from ...type import GraphQLSchema
from ..base import ExecutionContext, ExecutionResult, get_operation_root_type
from ..executors.sync import SyncExecutor
from ..middleware import MiddlewareManager
from .fragment import Fragment
def on_resolve(data):
    if not context.errors:
        return ExecutionResult(data=data)
    return ExecutionResult(data=data, errors=context.errors)