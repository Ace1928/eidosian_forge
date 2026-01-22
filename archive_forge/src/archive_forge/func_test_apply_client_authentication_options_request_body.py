import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test_apply_client_authentication_options_request_body(self):
    headers = {'Content-Type': 'application/json'}
    request_body = {'foo': 'bar'}
    auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_REQUEST_BODY)
    auth_handler.apply_client_authentication_options(headers, request_body)
    assert headers == {'Content-Type': 'application/json'}
    assert request_body == {'foo': 'bar', 'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}