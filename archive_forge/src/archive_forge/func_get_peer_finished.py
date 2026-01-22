import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_peer_finished(self):
    """
        Obtain the latest TLS Finished message that we received from the peer.

        :return: The contents of the message or :obj:`None` if the TLS
            handshake has not yet completed.
        :rtype: :class:`bytes` or :class:`NoneType`

        .. versionadded:: 0.15
        """
    return self._get_finished_message(_lib.SSL_get_peer_finished)