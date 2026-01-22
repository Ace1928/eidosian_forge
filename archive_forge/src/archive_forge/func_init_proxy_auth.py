import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def init_proxy_auth(self):
    self.proxy_requests = True
    self.auth_header_sent = 'Proxy-Authenticate'
    self.auth_header_recv = 'Proxy-Authorization'
    self.auth_error_code = 407