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
def batch_drop_table_comment(cls, operations: BatchOperations, *, existing_comment: Optional[str]=None) -> None:
    """Issue a "drop table comment" operation to
        remove an existing comment set on a table using the current
        batch operations context.

        :param existing_comment: An optional string value of a comment already
         registered on the specified table.

        """
    op = cls(operations.impl.table_name, existing_comment=existing_comment, schema=operations.impl.schema)
    return operations.invoke(op)