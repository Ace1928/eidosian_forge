import socket
import sys
import warnings
from . import base_events
from . import constants
from . import futures
from . import sslproto
from . import transports
from .log import logger
def sock_connect(self, sock, address):
    try:
        if self._debug:
            base_events._check_resolved_address(sock, address)
    except ValueError as err:
        fut = futures.Future(loop=self)
        fut.set_exception(err)
        return fut
    else:
        return self._proactor.connect(sock, address)