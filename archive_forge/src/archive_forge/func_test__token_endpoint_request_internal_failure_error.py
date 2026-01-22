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
def test__token_endpoint_request_internal_failure_error():
    request = make_request({'error_description': 'internal_failure'}, status=http_client.BAD_REQUEST)
    with pytest.raises(exceptions.RefreshError):
        _client._token_endpoint_request(request, 'http://example.com', {'error_description': 'internal_failure'})
    assert request.call_count == 4
    request = make_request({'error': 'internal_failure'}, status=http_client.BAD_REQUEST)
    with pytest.raises(exceptions.RefreshError):
        _client._token_endpoint_request(request, 'http://example.com', {'error': 'internal_failure'})
    assert request.call_count == 4