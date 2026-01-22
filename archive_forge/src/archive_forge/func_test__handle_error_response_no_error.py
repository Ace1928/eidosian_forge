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
def test__handle_error_response_no_error():
    response_data = {'foo': 'bar'}
    with pytest.raises(exceptions.RefreshError) as excinfo:
        _client._handle_error_response(response_data, False)
    assert not excinfo.value.retryable
    assert excinfo.match('{\\"foo\\": \\"bar\\"}')