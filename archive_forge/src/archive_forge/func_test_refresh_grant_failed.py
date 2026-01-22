import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test_refresh_grant_failed():
    with mock.patch('google.oauth2._client._token_endpoint_request_no_throw') as mock_token_request:
        mock_token_request.return_value = (False, {'error': 'Bad request'}, False)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            reauth.refresh_grant(MOCK_REQUEST, 'token_uri', 'refresh_token', 'client_id', 'client_secret', scopes=['foo', 'bar'], rapt_token='rapt_token', enable_reauth_refresh=True)
        assert excinfo.match('Bad request')
        assert not excinfo.value.retryable
        mock_token_request.assert_called_with(MOCK_REQUEST, 'token_uri', {'grant_type': 'refresh_token', 'client_id': 'client_id', 'client_secret': 'client_secret', 'refresh_token': 'refresh_token', 'scope': 'foo bar', 'rapt': 'rapt_token'})