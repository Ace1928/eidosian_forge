import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import exceptions
from google.auth import external_account_authorized_user
from google.auth import transport
@mock.patch('google.auth._helpers.utcnow', return_value=NOW)
def test_refresh_auth_success(self, utcnow):
    request = self.make_mock_request(status=http_client.OK, data={'access_token': ACCESS_TOKEN, 'expires_in': 3600})
    creds = self.make_credentials()
    creds.refresh(request)
    assert creds.expiry == utcnow() + datetime.timedelta(seconds=3600)
    assert not creds.expired
    assert creds.token == ACCESS_TOKEN
    assert creds.valid
    assert not creds.requires_scopes
    assert creds.is_user
    assert creds._refresh_token == REFRESH_TOKEN
    request.assert_called_once_with(url=TOKEN_URL, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Basic ' + BASIC_AUTH_ENCODING}, body=('grant_type=refresh_token&refresh_token=' + REFRESH_TOKEN).encode('UTF-8'))