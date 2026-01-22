from __future__ import annotations
import asyncio
import pickle
import warnings
from queue import Queue
from typing import Any, Awaitable, Callable, Sequence, cast, overload
from tornado.ioloop import IOLoop
from tornado.log import gen_log
import zmq
import zmq._future
from zmq import POLLIN, POLLOUT
from zmq._typing import Literal
from zmq.utils import jsonapi
def on_recv(self, callback: Callable[[list[zmq.Frame]], Any] | Callable[[list[bytes]], Any], copy: bool=True) -> None:
    """Register a callback for when a message is ready to recv.

        There can be only one callback registered at a time, so each
        call to `on_recv` replaces previously registered callbacks.

        on_recv(None) disables recv event polling.

        Use on_recv_stream(callback) instead, to register a callback that will receive
        both this ZMQStream and the message, instead of just the message.

        Parameters
        ----------

        callback : callable
            callback must take exactly one argument, which will be a
            list, as returned by socket.recv_multipart()
            if callback is None, recv callbacks are disabled.
        copy : bool
            copy is passed directly to recv, so if copy is False,
            callback will receive Message objects. If copy is True,
            then callback will receive bytes/str objects.

        Returns : None
        """
    self._check_closed()
    assert callback is None or callable(callback)
    self._recv_callback = callback
    self._recv_copy = copy
    if callback is None:
        self._drop_io_state(zmq.POLLIN)
    else:
        self._add_io_state(zmq.POLLIN)