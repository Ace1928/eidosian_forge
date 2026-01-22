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
def get_head_revision(self) -> _RevNumber:
    """Return the hex identifier of the 'head' script revision.

        If the script directory has multiple heads, this
        method raises a :class:`.CommandError`;
        :meth:`.EnvironmentContext.get_head_revisions` should be preferred.

        This function does not require that the :class:`.MigrationContext`
        has been configured.

        .. seealso:: :meth:`.EnvironmentContext.get_head_revisions`

        """
    return self.script.as_revision_number('head')