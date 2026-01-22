import datetime
import json
import os
import mock
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.auth.jwt.Credentials._make_jwt')
def test_refresh_with_jwt_credentials(self, make_jwt):
    credentials = self.make_credentials()
    credentials._create_self_signed_jwt('https://pubsub.googleapis.com')
    request = mock.create_autospec(transport.Request, instance=True)
    token = 'token'
    expiry = _helpers.utcnow() + datetime.timedelta(seconds=500)
    make_jwt.return_value = (token, expiry)
    assert not credentials.valid
    credentials.before_request(request, 'GET', 'http://example.com?a=1#3', {})
    assert credentials.valid
    assert make_jwt.call_count == 1
    assert credentials.token == token
    assert credentials.expiry == expiry