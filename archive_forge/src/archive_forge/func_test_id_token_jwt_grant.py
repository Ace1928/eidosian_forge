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
def test_id_token_jwt_grant():
    now = _helpers.utcnow()
    id_token_expiry = _helpers.datetime_to_secs(now)
    id_token = jwt.encode(SIGNER, {'exp': id_token_expiry}).decode('utf-8')
    request = make_request({'id_token': id_token, 'extra': 'data'})
    token, expiry, extra_data = _client.id_token_jwt_grant(request, 'http://example.com', 'assertion_value')
    verify_request_params(request, {'grant_type': _client._JWT_GRANT_TYPE, 'assertion': 'assertion_value'})
    assert token == id_token
    now = now.replace(microsecond=0)
    assert expiry == now
    assert extra_data['extra'] == 'data'