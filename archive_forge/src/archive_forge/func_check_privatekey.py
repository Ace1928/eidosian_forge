import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def check_privatekey(self):
    """
        Check if the private key (loaded with :meth:`use_privatekey`) matches
        the certificate (loaded with :meth:`use_certificate`)

        :return: :data:`None` (raises :exc:`Error` if something's wrong)
        """
    if not _lib.SSL_CTX_check_private_key(self._context):
        _raise_current_error()