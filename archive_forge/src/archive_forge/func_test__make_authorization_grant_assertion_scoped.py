import datetime
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _service_account_async as service_account
from tests.oauth2 import test_service_account
def test__make_authorization_grant_assertion_scoped(self):
    credentials = self.make_credentials()
    scopes = ['email', 'profile']
    credentials = credentials.with_scopes(scopes)
    token = credentials._make_authorization_grant_assertion()
    payload = jwt.decode(token, test_service_account.PUBLIC_CERT_BYTES)
    assert payload['scope'] == 'email profile'