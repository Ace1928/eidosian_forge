import asyncio
from concurrent import futures
import functools
import sys
import types
from tornado.log import app_log
import typing
from typing import Any, Callable, Optional, Tuple, Union
def future_set_exc_info(future: 'Union[futures.Future[_T], Future[_T]]', exc_info: Tuple[Optional[type], Optional[BaseException], Optional[types.TracebackType]]) -> None:
    """Set the given ``exc_info`` as the `Future`'s exception.

    Understands both `asyncio.Future` and the extensions in older
    versions of Tornado to enable better tracebacks on Python 2.

    .. versionadded:: 5.0

    .. versionchanged:: 6.0

       If the future is already cancelled, this function is a no-op.
       (previously ``asyncio.InvalidStateError`` would be raised)

    """
    if exc_info[1] is None:
        raise Exception('future_set_exc_info called with no exception')
    future_set_exception_unless_cancelled(future, exc_info[1])