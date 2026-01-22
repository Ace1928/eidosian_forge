import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import exceptions
from google.auth import external_account_authorized_user
from google.auth import transport
def test_refresh_auth_failure(self):
    request = self.make_mock_request(status=http_client.BAD_REQUEST, data={'error': 'invalid_request', 'error_description': 'Invalid subject token', 'error_uri': 'https://tools.ietf.org/html/rfc6749'})
    creds = self.make_credentials()
    with pytest.raises(exceptions.OAuthError) as excinfo:
        creds.refresh(request)
    assert excinfo.match('Error code invalid_request: Invalid subject token - https://tools.ietf.org/html/rfc6749')
    assert not creds.expiry
    assert not creds.expired
    assert not creds.token
    assert not creds.valid
    assert not creds.requires_scopes
    assert creds.is_user
    request.assert_called_once_with(url=TOKEN_URL, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Basic ' + BASIC_AUTH_ENCODING}, body=('grant_type=refresh_token&refresh_token=' + REFRESH_TOKEN).encode('UTF-8'))