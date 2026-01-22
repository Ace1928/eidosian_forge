import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def renegotiate(self):
    """
        Renegotiate the session.

        :return: True if the renegotiation can be started, False otherwise
        :rtype: bool
        """
    if not self.renegotiate_pending():
        _openssl_assert(_lib.SSL_renegotiate(self._ssl) == 1)
        return True
    return False