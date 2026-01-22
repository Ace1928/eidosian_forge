from __future__ import annotations
import abc
import functools
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Generator
from typing import Generic
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
import weakref
from . import exc as async_exc
from ... import util
from ...util.typing import Literal
from ...util.typing import Self
def asyncstartablecontext(func: Callable[..., AsyncIterator[_T_co]]) -> Callable[..., GeneratorStartableContext[_T_co]]:
    """@asyncstartablecontext decorator.

    the decorated function can be called either as ``async with fn()``, **or**
    ``await fn()``.   This is decidedly different from what
    ``@contextlib.asynccontextmanager`` supports, and the usage pattern
    is different as well.

    Typical usage::

        @asyncstartablecontext
        async def some_async_generator(<arguments>):
            <setup>
            try:
                yield <value>
            except GeneratorExit:
                # return value was awaited, no context manager is present
                # and caller will .close() the resource explicitly
                pass
            else:
                <context manager cleanup>


    Above, ``GeneratorExit`` is caught if the function were used as an
    ``await``.  In this case, it's essential that the cleanup does **not**
    occur, so there should not be a ``finally`` block.

    If ``GeneratorExit`` is not invoked, this means we're in ``__aexit__``
    and we were invoked as a context manager, and cleanup should proceed.


    """

    @functools.wraps(func)
    def helper(*args: Any, **kwds: Any) -> GeneratorStartableContext[_T_co]:
        return GeneratorStartableContext(func, args, kwds)
    return helper