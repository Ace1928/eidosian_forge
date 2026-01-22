import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
@mock.patch('google.oauth2.id_token.verify_token', autospec=True)
def test_verify_oauth2_token_invalid_iss(verify_token):
    verify_token.return_value = {'iss': 'invalid_issuer'}
    with pytest.raises(exceptions.GoogleAuthError):
        id_token.verify_oauth2_token(mock.sentinel.token, mock.sentinel.request, audience=mock.sentinel.audience)