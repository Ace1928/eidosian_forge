from __future__ import annotations
import errno
import pickle
import random
import sys
from typing import (
from warnings import warn
import zmq
from zmq._typing import Literal, TypeAlias
from zmq.backend import Socket as SocketBase
from zmq.error import ZMQBindError, ZMQError
from zmq.utils import jsonapi
from zmq.utils.interop import cast_int_addr
from ..constants import SocketOption, SocketType, _OptType
from .attrsettr import AttributeSetter
from .poll import Poller
def recv_pyobj(self, flags: int=0) -> Any:
    """Receive a Python object as a message using pickle to serialize.

        Parameters
        ----------
        flags : int
            Any valid flags for :func:`Socket.recv`.

        Returns
        -------
        obj : Python object
            The Python object that arrives as a message.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
    msg = self.recv(flags)
    return self._deserialize(msg, pickle.loads)