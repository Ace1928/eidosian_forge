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
@pytest.mark.parametrize('can_retry', [True, False])
@mock.patch('google.oauth2._client._parse_expiry', return_value=None)
@mock.patch.object(_client, '_token_endpoint_request', autospec=True)
def test_refresh_grant_retry_with_retry(mock_token_endpoint_request, mock_parse_expiry, can_retry):
    _client.refresh_grant(mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock(), can_retry=can_retry)
    mock_token_endpoint_request.assert_called_with(mock.ANY, mock.ANY, mock.ANY, can_retry=can_retry)