import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import transport
from google.auth.impersonated_credentials import Credentials
from google.oauth2 import credentials
from google.oauth2 import service_account
@pytest.fixture
def mock_authorizedsession_idtoken():
    with mock.patch('google.auth.transport.requests.AuthorizedSession.request', autospec=True) as auth_session:
        data = {'token': ID_TOKEN_DATA}
        auth_session.return_value = MockResponse(data, http_client.OK)
        yield auth_session