import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test_get_rapt_token():
    with mock.patch('google.oauth2._client.refresh_grant', return_value=('token', None, None, None)) as mock_refresh_grant:
        with mock.patch('google.oauth2.reauth._obtain_rapt', return_value='new_rapt_token') as mock_obtain_rapt:
            assert reauth.get_rapt_token(MOCK_REQUEST, 'client_id', 'client_secret', 'refresh_token', 'token_uri') == 'new_rapt_token'
            mock_refresh_grant.assert_called_with(request=MOCK_REQUEST, client_id='client_id', client_secret='client_secret', refresh_token='refresh_token', token_uri='token_uri', scopes=[reauth._REAUTH_SCOPE])
            mock_obtain_rapt.assert_called_with(MOCK_REQUEST, 'token', requested_scopes=None)