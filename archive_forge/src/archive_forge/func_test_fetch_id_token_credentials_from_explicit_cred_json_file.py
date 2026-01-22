import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
def test_fetch_id_token_credentials_from_explicit_cred_json_file(monkeypatch):
    monkeypatch.setenv(environment_vars.CREDENTIALS, SERVICE_ACCOUNT_FILE)
    cred = id_token.fetch_id_token_credentials(ID_TOKEN_AUDIENCE)
    assert isinstance(cred, service_account.IDTokenCredentials)
    assert cred._target_audience == ID_TOKEN_AUDIENCE