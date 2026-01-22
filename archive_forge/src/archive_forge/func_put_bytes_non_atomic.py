import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def put_bytes_non_atomic(self, relpath, bytes: bytes, mode=None, create_parent_dir=False, dir_mode=False):
    """See Transport.put_file_non_atomic"""
    abspath = self._remote_path(relpath)
    headers = {'Accept': '*/*', 'Content-type': 'application/octet-stream'}

    def bare_put_file_non_atomic():
        response = self.request('PUT', abspath, body=bytes, headers=headers)
        code = response.status
        if code in (403, 404, 409):
            raise transport.NoSuchFile(abspath)
        elif code not in (200, 201, 204):
            raise self._raise_http_error(abspath, response, 'put file failed')
    try:
        bare_put_file_non_atomic()
    except transport.NoSuchFile:
        if not create_parent_dir:
            raise
        parent_dir = osutils.dirname(relpath)
        if parent_dir:
            self.mkdir(parent_dir, mode=dir_mode)
            return bare_put_file_non_atomic()
        else:
            raise