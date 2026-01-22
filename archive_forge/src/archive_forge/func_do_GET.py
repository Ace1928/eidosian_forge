import base64
import datetime
import json
import platform
import threading
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import pymacaroons
import requests
import macaroonbakery._utils as utils
from macaroonbakery.httpbakery._error import DischargeError
from fixtures import (
from httmock import HTTMock, urlmatch
from six.moves.urllib.parse import parse_qs
from six.moves.urllib.request import Request
def do_GET(self):
    """do_GET implements a handler for the HTTP GET method"""
    ctx = checkers.AuthContext()
    auth_checker = self._bakery.checker.auth(httpbakery.extract_macaroons(self.headers))
    try:
        auth_checker.allow(ctx, [TEST_OP])
    except (bakery.PermissionDenied, bakery.VerificationError) as exc:
        return self._write_discharge_error(exc)
    self.send_response(200)
    self.end_headers()
    content_len = int(self.headers.get('content-length', 0))
    content = 'done'
    if self.path != '/no-body' and content_len > 0:
        body = self.rfile.read(content_len)
        content = content + ' ' + body
    self.wfile.write(content.encode('utf-8'))
    return