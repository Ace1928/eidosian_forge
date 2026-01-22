import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
def test_fetch_id_token_credentials_optional_request(monkeypatch):
    monkeypatch.delenv(environment_vars.CREDENTIALS, raising=False)
    with mock.patch('google.auth.compute_engine._metadata.ping', return_value=True):
        with mock.patch('google.auth.compute_engine.IDTokenCredentials.__init__', return_value=None):
            with mock.patch('google.auth.transport.requests.Request.__init__', return_value=None) as mock_request:
                id_token.fetch_id_token_credentials(ID_TOKEN_AUDIENCE)
            mock_request.assert_called()