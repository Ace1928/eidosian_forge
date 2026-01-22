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
def test_cookie_domain_host_not_fqdn(self):
    b = new_bakery('loc', None, None)

    def handler(*args):
        GetHandler(b, None, None, None, None, AGES, *args)
    try:
        httpd = HTTPServer(('', 0), handler)
        thread = threading.Thread(target=httpd.serve_forever)
        thread.start()
        srv_macaroon = b.oven.macaroon(version=bakery.LATEST_VERSION, expiry=AGES, caveats=None, ops=[TEST_OP])
        self.assertEqual(srv_macaroon.macaroon.location, 'loc')
        client = httpbakery.Client()
        resp = requests.get(url='http://localhost:' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
        resp.raise_for_status()
        self.assertEqual(resp.text, 'done')
    except httpbakery.BakeryException:
        pass
    finally:
        httpd.shutdown()
    [cookie] = client.cookies
    self.assertEqual(cookie.name, 'macaroon-test')
    self.assertEqual(cookie.domain, 'localhost.local')