import datetime
import json
import os
import mock
import pytest  # type: ignore
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _client
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test_jwt_grant(utcnow):
    request = make_request({'access_token': 'token', 'expires_in': 500, 'extra': 'data'})
    token, expiry, extra_data = _client.jwt_grant(request, 'http://example.com', 'assertion_value')
    verify_request_params(request, {'grant_type': _client._JWT_GRANT_TYPE, 'assertion': 'assertion_value'})
    assert token == 'token'
    assert expiry == utcnow() + datetime.timedelta(seconds=500)
    assert extra_data['extra'] == 'data'