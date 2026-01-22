import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_initial_call_to_get_access_token(self):
    """Test initial call, expect JSON access token."""
    self.requests_mock.post(self.ACCESS_TOKEN_ENDPOINT, json=oidc_fixtures.ACCESS_TOKEN_VIA_AUTH_GRANT_RESP)
    grant_type = 'authorization_code'
    payload = {'grant_type': grant_type, 'redirect_uri': self.REDIRECT_URL, 'code': self.CODE}
    self.plugin._get_access_token(self.session, payload)
    last_req = self.requests_mock.last_request
    self.assertEqual(self.ACCESS_TOKEN_ENDPOINT, last_req.url)
    self.assertEqual('POST', last_req.method)
    encoded_payload = urllib.parse.urlencode(payload)
    self.assertEqual(encoded_payload, last_req.body)