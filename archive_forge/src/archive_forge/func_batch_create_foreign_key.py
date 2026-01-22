from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
@classmethod
def batch_create_foreign_key(cls, operations: BatchOperations, constraint_name: str, referent_table: str, local_cols: List[str], remote_cols: List[str], *, referent_schema: Optional[str]=None, onupdate: Optional[str]=None, ondelete: Optional[str]=None, deferrable: Optional[bool]=None, initially: Optional[str]=None, match: Optional[str]=None, **dialect_kw: Any) -> None:
    """Issue a "create foreign key" instruction using the
        current batch migration context.

        The batch form of this call omits the ``source`` and ``source_schema``
        arguments from the call.

        e.g.::

            with batch_alter_table("address") as batch_op:
                batch_op.create_foreign_key(
                    "fk_user_address",
                    "user",
                    ["user_id"],
                    ["id"],
                )

        .. seealso::

            :meth:`.Operations.create_foreign_key`

        """
    op = cls(constraint_name, operations.impl.table_name, referent_table, local_cols, remote_cols, onupdate=onupdate, ondelete=ondelete, deferrable=deferrable, source_schema=operations.impl.schema, referent_schema=referent_schema, initially=initially, match=match, **dialect_kw)
    return operations.invoke(op)