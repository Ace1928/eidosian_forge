from __future__ import annotations
import atexit
import os
from threading import Lock
from typing import Any, Callable, Generic, TypeVar, overload
from warnings import warn
from weakref import WeakSet
import zmq
from zmq._typing import TypeAlias
from zmq.backend import Context as ContextBase
from zmq.constants import ContextOption, Errno, SocketOption
from zmq.error import ZMQError
from zmq.utils.interop import cast_int_addr
from .attrsettr import AttributeSetter, OptValT
from .socket import Socket, SyncSocket
@classmethod
def shadow_pyczmq(cls: type[_ContextType], ctx: Any) -> _ContextType:
    """Shadow an existing pyczmq context

        ctx is the FFI `zctx_t *` pointer

        .. versionadded:: 14.1
        """
    from pyczmq import zctx
    from zmq.utils.interop import cast_int_addr
    underlying = zctx.underlying(ctx)
    address = cast_int_addr(underlying)
    return cls(shadow=address)