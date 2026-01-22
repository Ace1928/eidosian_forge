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
def set_string(self, option: int, optval: str, encoding='utf-8') -> None:
    """Set socket options with a unicode object.

        This is simply a wrapper for setsockopt to protect from encoding ambiguity.

        See the 0MQ documentation for details on specific options.

        Parameters
        ----------
        option : int
            The name of the option to set. Can be any of: SUBSCRIBE,
            UNSUBSCRIBE, IDENTITY
        optval : str
            The value of the option to set.
        encoding : str
            The encoding to be used, default is utf8
        """
    if not isinstance(optval, str):
        raise TypeError(f'strings only, not {type(optval)}: {optval!r}')
    return self.set(option, optval.encode(encoding))