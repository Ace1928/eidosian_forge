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
def get_tag_argument(self) -> Optional[str]:
    """Return the value passed for the ``--tag`` argument, if any.

        The ``--tag`` argument is not used directly by Alembic,
        but is available for custom ``env.py`` configurations that
        wish to use it; particularly for offline generation scripts
        that wish to generate tagged filenames.

        This function does not require that the :class:`.MigrationContext`
        has been configured.

        .. seealso::

            :meth:`.EnvironmentContext.get_x_argument` - a newer and more
            open ended system of extending ``env.py`` scripts via the command
            line.

        """
    return self.context_opts.get('tag', None)