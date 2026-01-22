import errno
import os
import re
import shutil  # FIXME: Can't we use breezy.osutils ?
import stat
import time
import urllib.parse  # FIXME: Can't we use breezy.urlutils ?
from breezy import trace, urlutils
from breezy.tests import http_server
def read_body(self):
    """Read the body either by chunk or as a whole."""
    content_length = self.headers.get('Content-Length')
    encoding = self.headers.get('Transfer-Encoding')
    if encoding is not None:
        assert encoding == 'chunked'
        body = []
        while True:
            length, data = self.read_chunk()
            if length == 0:
                break
            body.append(data)
        body = ''.join(body)
    elif content_length is not None:
        body = self._read(int(content_length))
    return body