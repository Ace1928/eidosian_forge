import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
def test_fetch_id_token_credentials_no_cred_exists(monkeypatch):
    monkeypatch.delenv(environment_vars.CREDENTIALS, raising=False)
    with mock.patch('google.auth.compute_engine._metadata.ping', side_effect=exceptions.TransportError()):
        with pytest.raises(exceptions.DefaultCredentialsError) as excinfo:
            id_token.fetch_id_token_credentials(ID_TOKEN_AUDIENCE)
        assert excinfo.match('Neither metadata server or valid service account credentials are found.')
    with mock.patch('google.auth.compute_engine._metadata.ping', return_value=False):
        with pytest.raises(exceptions.DefaultCredentialsError) as excinfo:
            id_token.fetch_id_token_credentials(ID_TOKEN_AUDIENCE)
        assert excinfo.match('Neither metadata server or valid service account credentials are found.')