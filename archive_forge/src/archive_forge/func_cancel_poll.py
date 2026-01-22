from __future__ import annotations
import warnings
from asyncio import Future
from collections import deque
from functools import partial
from itertools import chain
from typing import Any, Awaitable, Callable, NamedTuple, TypeVar, cast, overload
import zmq as _zmq
from zmq import EVENTS, POLLIN, POLLOUT
from zmq._typing import Literal
def cancel_poll(future):
    """Cancel underlying poll if request has been cancelled"""
    if not poll_future.done():
        try:
            poll_future.cancel()
        except RuntimeError:
            pass