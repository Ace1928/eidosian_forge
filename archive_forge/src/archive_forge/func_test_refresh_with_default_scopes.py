import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test_refresh_with_default_scopes(self, utcnow, app_identity):
    token = 'token'
    ttl = 643942923
    app_identity.get_access_token.return_value = (token, ttl)
    credentials = app_engine.Credentials(default_scopes=['email'])
    credentials.refresh(None)
    app_identity.get_access_token.assert_called_with(credentials.default_scopes, credentials._service_account_id)
    assert credentials.token == token
    assert credentials.expiry == datetime.datetime(1990, 5, 29, 1, 2, 3)
    assert credentials.valid
    assert not credentials.expired