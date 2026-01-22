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
@pytest.mark.parametrize('mock_expires_in', [500, '500'])
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.min)
def test__parse_expiry(unused_utcnow, mock_expires_in):
    result = _client._parse_expiry({'expires_in': mock_expires_in})
    assert result == datetime.datetime.min + datetime.timedelta(seconds=500)