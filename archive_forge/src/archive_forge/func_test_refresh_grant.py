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
def test_refresh_grant(unused_utcnow):
    request = make_request({'access_token': 'token', 'refresh_token': 'new_refresh_token', 'expires_in': 500, 'extra': 'data'})
    token, refresh_token, expiry, extra_data = _client.refresh_grant(request, 'http://example.com', 'refresh_token', 'client_id', 'client_secret', rapt_token='rapt_token')
    verify_request_params(request, {'grant_type': _client._REFRESH_GRANT_TYPE, 'refresh_token': 'refresh_token', 'client_id': 'client_id', 'client_secret': 'client_secret', 'rapt': 'rapt_token'})
    assert token == 'token'
    assert refresh_token == 'new_refresh_token'
    assert expiry == datetime.datetime.min + datetime.timedelta(seconds=500)
    assert extra_data['extra'] == 'data'