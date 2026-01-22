from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
def test_discharge(self):
    client = httpbakery.Client()
    with HTTMock(first_407_then_200), HTTMock(discharge_200):
        resp = requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
    resp.raise_for_status()
    assert 'macaroon-test' in client.cookies.keys()
    self.assert_cookie_security(client.cookies, 'macaroon-test', secure=False)