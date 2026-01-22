from __future__ import annotations
from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict
def update_url(self, url: URL) -> URL:
    """Update the :class:`_engine.URL`.

        A new :class:`_engine.URL` should be returned.   This method is
        typically used to consume configuration arguments from the
        :class:`_engine.URL` which must be removed, as they will not be
        recognized by the dialect.  The
        :meth:`_engine.URL.difference_update_query` method is available
        to remove these arguments.   See the docstring at
        :class:`_engine.CreateEnginePlugin` for an example.


        .. versionadded:: 1.4

        """
    raise NotImplementedError()