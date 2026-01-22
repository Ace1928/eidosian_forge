from typing import TYPE_CHECKING
from sqlalchemy import schema as sa_schema
from . import ops
from .base import Operations
from ..util.sqla_compat import _copy
from ..util.sqla_compat import sqla_14
@Operations.implementation_for(ops.BulkInsertOp)
def bulk_insert(operations: 'Operations', operation: 'ops.BulkInsertOp') -> None:
    operations.impl.bulk_insert(operation.table, operation.rows, multiinsert=operation.multiinsert)