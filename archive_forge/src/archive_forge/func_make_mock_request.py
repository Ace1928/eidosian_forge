import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
@classmethod
def make_mock_request(cls, data, status=http_client.OK):
    response = mock.create_autospec(transport.Response, instance=True)
    response.status = status
    response.data = json.dumps(data).encode('utf-8')
    request = mock.create_autospec(transport.Request)
    request.return_value = response
    return request