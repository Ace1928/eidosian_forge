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
def test_default_state(self):
    credentials = self.make_credentials()
    assert not credentials.valid
    assert not credentials.expired
    assert not credentials.requires_scopes
    assert credentials.refresh_token == self.REFRESH_TOKEN
    assert credentials.token_uri == self.TOKEN_URI
    assert credentials.client_id == self.CLIENT_ID
    assert credentials.client_secret == self.CLIENT_SECRET