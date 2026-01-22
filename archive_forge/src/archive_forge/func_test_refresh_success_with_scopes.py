import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min + _helpers.REFRESH_THRESHOLD)
@mock.patch('google.auth.compute_engine._metadata.get', autospec=True)
def test_refresh_success_with_scopes(self, get, utcnow):
    get.side_effect = [{'email': 'service-account@example.com', 'scopes': ['one', 'two']}, {'access_token': 'token', 'expires_in': 500}]
    scopes = ['three', 'four']
    self.credentials = self.credentials.with_scopes(scopes)
    self.credentials.refresh(None)
    assert self.credentials.token == 'token'
    assert self.credentials.expiry == utcnow() + datetime.timedelta(seconds=500)
    assert self.credentials.service_account_email == 'service-account@example.com'
    assert self.credentials._scopes == scopes
    assert self.credentials.valid
    kwargs = get.call_args[1]
    assert kwargs == {'params': {'scopes': 'three,four'}}