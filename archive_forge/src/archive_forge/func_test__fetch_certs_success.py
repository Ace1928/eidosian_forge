import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
def test__fetch_certs_success():
    certs = {'1': 'cert'}
    request = make_request(200, certs)
    returned_certs = id_token._fetch_certs(request, mock.sentinel.cert_url)
    request.assert_called_once_with(mock.sentinel.cert_url, method='GET')
    assert returned_certs == certs