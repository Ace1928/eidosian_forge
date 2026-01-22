import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
def handle_one_request(self):
    """Handle a single HTTP request.

        We catch all socket errors occurring when the client close the
        connection early to avoid polluting the test results.
        """
    try:
        self._handle_one_request()
    except OSError as e:
        self.close_connection = 1
        if len(e.args) == 0 or e.args[0] not in (errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.EBADF):
            raise