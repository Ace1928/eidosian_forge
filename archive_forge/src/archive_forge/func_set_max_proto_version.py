import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_max_proto_version(self, version):
    """
        Set the maximum supported protocol version. Setting the maximum
        version to 0 will enable protocol versions up to the highest version
        supported by the library.

        If the underlying OpenSSL build is missing support for the selected
        version, this method will raise an exception.
        """
    _openssl_assert(_lib.SSL_CTX_set_max_proto_version(self._context, version) == 1)