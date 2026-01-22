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
def test_id_token_jwt_grant_no_access_token():
    request = make_request({'expires_in': 500, 'extra': 'data'})
    with pytest.raises(exceptions.RefreshError) as excinfo:
        _client.id_token_jwt_grant(request, 'http://example.com', 'assertion_value')
    assert not excinfo.value.retryable