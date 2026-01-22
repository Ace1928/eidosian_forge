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
def get_revision_argument(self) -> _RevNumber:
    """Get the 'destination' revision argument.

        This is typically the argument passed to the
        ``upgrade`` or ``downgrade`` command.

        If it was specified as ``head``, the actual
        version number is returned; if specified
        as ``base``, ``None`` is returned.

        This function does not require that the :class:`.MigrationContext`
        has been configured.

        """
    return self.script.as_revision_number(self.context_opts['destination_rev'])