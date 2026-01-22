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
def test_call_iam_generate_id_token_endpoint_no_id_token():
    request = make_request({'error': 'no token'})
    with pytest.raises(exceptions.RefreshError) as excinfo:
        _client.call_iam_generate_id_token_endpoint(request, 'fake_email', 'fake_audience', 'fake_access_token')
    assert excinfo.match('No ID token in response')