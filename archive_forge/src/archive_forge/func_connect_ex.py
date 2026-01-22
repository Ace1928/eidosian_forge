import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def connect_ex(self, addr):
    """
        Call the :meth:`connect_ex` method of the underlying socket and set up
        SSL on the socket, using the Context object supplied to this Connection
        object at creation. Note that if the :meth:`connect_ex` method of the
        socket doesn't return 0, SSL won't be initialized.

        :param addr: A remove address
        :return: What the socket's connect_ex method returns
        """
    connect_ex = self._socket.connect_ex
    self.set_connect_state()
    return connect_ex(addr)