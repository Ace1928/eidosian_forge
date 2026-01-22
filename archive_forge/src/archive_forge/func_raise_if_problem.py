import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def raise_if_problem(self):
    """
        Raise an exception from the OpenSSL error queue or that was previously
        captured whe running a callback.
        """
    if self._problems:
        try:
            _raise_current_error()
        except Error:
            pass
        raise self._problems.pop(0)