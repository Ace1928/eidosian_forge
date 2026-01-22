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
def create_async_engine(url: Union[str, URL], **kw: Any) -> AsyncEngine:
    """Create a new async engine instance.

    Arguments passed to :func:`_asyncio.create_async_engine` are mostly
    identical to those passed to the :func:`_sa.create_engine` function.
    The specified dialect must be an asyncio-compatible dialect
    such as :ref:`dialect-postgresql-asyncpg`.

    .. versionadded:: 1.4

    :param async_creator: an async callable which returns a driver-level
        asyncio connection. If given, the function should take no arguments,
        and return a new asyncio connection from the underlying asyncio
        database driver; the connection will be wrapped in the appropriate
        structures to be used with the :class:`.AsyncEngine`.   Note that the
        parameters specified in the URL are not applied here, and the creator
        function should use its own connection parameters.

        This parameter is the asyncio equivalent of the
        :paramref:`_sa.create_engine.creator` parameter of the
        :func:`_sa.create_engine` function.

        .. versionadded:: 2.0.16

    """
    if kw.get('server_side_cursors', False):
        raise async_exc.AsyncMethodRequired("Can't set server_side_cursors for async engine globally; use the connection.stream() method for an async streaming result set")
    kw['_is_async'] = True
    async_creator = kw.pop('async_creator', None)
    if async_creator:
        if kw.get('creator', None):
            raise ArgumentError("Can only specify one of 'async_creator' or 'creator', not both.")

        def creator() -> Any:
            return sync_engine.dialect.dbapi.connect(async_creator_fn=async_creator)
        kw['creator'] = creator
    sync_engine = _create_engine(url, **kw)
    return AsyncEngine(sync_engine)