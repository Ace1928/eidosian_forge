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
@mock.patch('google.oauth2._credentials_async.UserAccessTokenCredentials.apply', autospec=True)
@mock.patch('google.oauth2._credentials_async.UserAccessTokenCredentials.refresh', autospec=True)
def test_before_request(self, refresh, apply):
    cred = _credentials_async.UserAccessTokenCredentials()
    cred.before_request(mock.Mock(), 'GET', 'https://example.com', {})
    refresh.assert_called()
    apply.assert_called()