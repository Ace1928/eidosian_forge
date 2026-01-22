from __future__ import annotations
import asyncio
import contextlib
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Dict
from typing import Generator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as async_exc
from .base import asyncstartablecontext
from .base import GeneratorStartableContext
from .base import ProxyComparable
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import exc
from ... import inspection
from ... import util
from ...engine import Connection
from ...engine import create_engine as _create_engine
from ...engine import create_pool_from_url as _create_pool_from_url
from ...engine import Engine
from ...engine.base import NestedTransaction
from ...engine.base import Transaction
from ...exc import ArgumentError
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
def get_nested_transaction(self) -> Optional[AsyncTransaction]:
    """Return an :class:`.AsyncTransaction` representing the current
        nested (savepoint) transaction, if any.

        This makes use of the underlying synchronous connection's
        :meth:`_engine.Connection.get_nested_transaction` method to get the
        current :class:`_engine.Transaction`, which is then proxied in a new
        :class:`.AsyncTransaction` object.

        .. versionadded:: 1.4.0b2

        """
    trans = self._proxied.get_nested_transaction()
    if trans is not None:
        return AsyncTransaction._retrieve_proxy_for_target(trans)
    else:
        return None