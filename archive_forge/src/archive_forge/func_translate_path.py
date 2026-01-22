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
def translate_path(self, path):
    """Translate a /-separated PATH to the local filename syntax.

        If the server requires it, proxy the path before the usual translation
        """
    if self.server.test_case_server.proxy_requests:
        path = urlparse(path)[2]
        path += '-proxied'
    return self._translate_path(path)