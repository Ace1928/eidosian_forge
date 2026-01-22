import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _credentials_async as _credentials_async
from google.oauth2 import credentials
from tests.oauth2 import test_credentials
def test_from_authorized_user_file(self):
    info = test_credentials.AUTH_USER_INFO.copy()
    creds = _credentials_async.Credentials.from_authorized_user_file(test_credentials.AUTH_USER_JSON_FILE)
    assert creds.client_secret == info['client_secret']
    assert creds.client_id == info['client_id']
    assert creds.refresh_token == info['refresh_token']
    assert creds.token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
    assert creds.scopes is None
    scopes = ['email', 'profile']
    creds = _credentials_async.Credentials.from_authorized_user_file(test_credentials.AUTH_USER_JSON_FILE, scopes)
    assert creds.client_secret == info['client_secret']
    assert creds.client_id == info['client_id']
    assert creds.refresh_token == info['refresh_token']
    assert creds.token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
    assert creds.scopes == scopes