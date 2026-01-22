import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import credentials
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test_refresh_with_refresh_handler_invalid_token(self, unused_utcnow):
    expected_expiry = datetime.datetime.min + datetime.timedelta(seconds=2800)
    refresh_handler = mock.Mock(return_value=(None, expected_expiry))
    scopes = ['email', 'profile']
    default_scopes = ['https://www.googleapis.com/auth/cloud-platform']
    request = mock.create_autospec(transport.Request)
    creds = credentials.Credentials(token=None, refresh_token=None, token_uri=None, client_id=None, client_secret=None, rapt_token=None, scopes=scopes, default_scopes=default_scopes, refresh_handler=refresh_handler)
    with pytest.raises(exceptions.RefreshError, match='returned token is not a string'):
        creds.refresh(request)
    assert creds.token is None
    assert creds.expiry is None
    assert not creds.valid
    refresh_handler.assert_called_with(request, scopes=scopes)