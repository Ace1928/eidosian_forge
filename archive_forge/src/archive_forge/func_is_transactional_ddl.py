from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.schema import FetchedValue
from typing_extensions import Literal
from .migration import _ProxyTransaction
from .migration import MigrationContext
from .. import util
from ..operations import Operations
from ..script.revision import _GetRevArg
def is_transactional_ddl(self) -> bool:
    """Return True if the context is configured to expect a
        transactional DDL capable backend.

        This defaults to the type of database in use, and
        can be overridden by the ``transactional_ddl`` argument
        to :meth:`.configure`

        This function requires that a :class:`.MigrationContext`
        has first been made available via :meth:`.configure`.

        """
    return self.get_context().impl.transactional_ddl