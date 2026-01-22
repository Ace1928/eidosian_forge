import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def parse_request(self):
    """Redirect a single HTTP request to another host"""
    valid = http_server.TestingHTTPRequestHandler.parse_request(self)
    if valid:
        tcs = self.server.test_case_server
        code, target = tcs.is_redirected(self.path)
        if code is not None and target is not None:
            self.send_response(code)
            self.send_header('Location', target)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return False
        else:
            pass
    return valid