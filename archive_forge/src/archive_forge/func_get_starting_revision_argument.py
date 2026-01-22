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
def get_starting_revision_argument(self) -> _RevNumber:
    """Return the 'starting revision' argument,
        if the revision was passed using ``start:end``.

        This is only meaningful in "offline" mode.
        Returns ``None`` if no value is available
        or was configured.

        This function does not require that the :class:`.MigrationContext`
        has been configured.

        """
    if self._migration_context is not None:
        return self.script.as_revision_number(self.get_context()._start_from_rev)
    elif 'starting_rev' in self.context_opts:
        return self.script.as_revision_number(self.context_opts['starting_rev'])
    else:
        raise util.CommandError('No starting revision argument is available.')