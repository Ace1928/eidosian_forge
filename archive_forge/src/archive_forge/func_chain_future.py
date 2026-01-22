import asyncio
from concurrent import futures
import functools
import sys
import types
from tornado.log import app_log
import typing
from typing import Any, Callable, Optional, Tuple, Union
def chain_future(a: 'Future[_T]', b: 'Future[_T]') -> None:
    """Chain two futures together so that when one completes, so does the other.

    The result (success or failure) of ``a`` will be copied to ``b``, unless
    ``b`` has already been completed or cancelled by the time ``a`` finishes.

    .. versionchanged:: 5.0

       Now accepts both Tornado/asyncio `Future` objects and
       `concurrent.futures.Future`.

    """

    def copy(a: 'Future[_T]') -> None:
        if b.done():
            return
        if hasattr(a, 'exc_info') and a.exc_info() is not None:
            future_set_exc_info(b, a.exc_info())
        else:
            a_exc = a.exception()
            if a_exc is not None:
                b.set_exception(a_exc)
            else:
                b.set_result(a.result())
    if isinstance(a, Future):
        future_add_done_callback(a, copy)
    else:
        from tornado.ioloop import IOLoop
        IOLoop.current().add_future(a, copy)