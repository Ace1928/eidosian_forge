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
def test_from_authorized_user_file_with_rapt_token(self):
    info = AUTH_USER_INFO.copy()
    file_path = os.path.join(DATA_DIR, 'authorized_user_with_rapt_token.json')
    creds = credentials.Credentials.from_authorized_user_file(file_path)
    assert creds.client_secret == info['client_secret']
    assert creds.client_id == info['client_id']
    assert creds.refresh_token == info['refresh_token']
    assert creds.token_uri == credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
    assert creds.scopes is None
    assert creds.rapt_token == 'rapt'