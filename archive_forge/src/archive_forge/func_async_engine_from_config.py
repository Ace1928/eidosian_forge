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
def async_engine_from_config(configuration: Dict[str, Any], prefix: str='sqlalchemy.', **kwargs: Any) -> AsyncEngine:
    """Create a new AsyncEngine instance using a configuration dictionary.

    This function is analogous to the :func:`_sa.engine_from_config` function
    in SQLAlchemy Core, except that the requested dialect must be an
    asyncio-compatible dialect such as :ref:`dialect-postgresql-asyncpg`.
    The argument signature of the function is identical to that
    of :func:`_sa.engine_from_config`.

    .. versionadded:: 1.4.29

    """
    options = {key[len(prefix):]: value for key, value in configuration.items() if key.startswith(prefix)}
    options['_coerce_config'] = True
    options.update(kwargs)
    url = options.pop('url')
    return create_async_engine(url, **options)