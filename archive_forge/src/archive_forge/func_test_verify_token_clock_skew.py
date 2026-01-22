import json
import os
import mock
import pytest  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import id_token
from google.oauth2 import service_account
@mock.patch('google.auth.jwt.decode', autospec=True)
@mock.patch('google.oauth2.id_token._fetch_certs', autospec=True)
def test_verify_token_clock_skew(_fetch_certs, decode):
    result = id_token.verify_token(mock.sentinel.token, mock.sentinel.request, audience=mock.sentinel.audience, certs_url=mock.sentinel.certs_url, clock_skew_in_seconds=10)
    assert result == decode.return_value
    _fetch_certs.assert_called_once_with(mock.sentinel.request, mock.sentinel.certs_url)
    decode.assert_called_once_with(mock.sentinel.token, certs=_fetch_certs.return_value, audience=mock.sentinel.audience, clock_skew_in_seconds=10)