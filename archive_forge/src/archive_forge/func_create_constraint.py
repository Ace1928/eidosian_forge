from typing import TYPE_CHECKING
from sqlalchemy import schema as sa_schema
from . import ops
from .base import Operations
from ..util.sqla_compat import _copy
from ..util.sqla_compat import sqla_14
@Operations.implementation_for(ops.AddConstraintOp)
def create_constraint(operations: 'Operations', operation: 'ops.AddConstraintOp') -> None:
    operations.impl.add_constraint(operation.to_constraint(operations.migration_context))