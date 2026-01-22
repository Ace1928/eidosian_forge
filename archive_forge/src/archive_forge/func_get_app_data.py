import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_app_data(self):
    """
        Retrieve application data as set by :meth:`set_app_data`.

        :return: The application data
        """
    return self._app_data