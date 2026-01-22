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
def test_refresh_handler_setter_and_getter(self):
    scopes = ['email', 'profile']
    original_refresh_handler = mock.Mock(return_value=('ACCESS_TOKEN_1', None))
    updated_refresh_handler = mock.Mock(return_value=('ACCESS_TOKEN_2', None))
    creds = credentials.Credentials(token=None, refresh_token=None, token_uri=None, client_id=None, client_secret=None, rapt_token=None, scopes=scopes, default_scopes=None, refresh_handler=original_refresh_handler)
    assert creds.refresh_handler is original_refresh_handler
    creds.refresh_handler = updated_refresh_handler
    assert creds.refresh_handler is updated_refresh_handler
    creds.refresh_handler = None
    assert creds.refresh_handler is None