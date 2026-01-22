from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
def update_response(self, response, result):
    response.headers = self.get_headers(result)
    response._content = result.read()
    response.status = result.getcode()
    response.url = result.geturl()
    response.msg = 'OK (%s bytes)' % response.headers.get('Content-Length', 'unknown')