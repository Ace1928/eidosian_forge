from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def set_persistent(self, on=True, auto_open=False):
    """Make our HTTP_CONN persistent (or not).

        If the 'on' argument is True (the default), then self.HTTP_CONN
        will be set to an instance of HTTP(S)?Connection
        to persist across requests.
        As this class only allows for a single open connection, if
        self already has an open connection, it will be closed.
        """
    try:
        self.HTTP_CONN.close()
    except (TypeError, AttributeError):
        pass
    self.HTTP_CONN = self.get_conn(auto_open=auto_open) if on else self._Conn