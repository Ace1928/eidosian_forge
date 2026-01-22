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
def disable_monitor(self) -> None:
    """Shutdown the PAIR socket (created using get_monitor_socket)
        that is serving socket events.

        .. versionadded:: 14.4
        """
    self._monitor_socket = None
    self.monitor(None, 0)