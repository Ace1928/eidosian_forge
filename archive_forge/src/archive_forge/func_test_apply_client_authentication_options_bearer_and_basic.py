import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test_apply_client_authentication_options_bearer_and_basic(self):
    bearer_token = 'ACCESS_TOKEN'
    headers = {'Content-Type': 'application/json'}
    request_body = {'foo': 'bar'}
    auth_handler = self.make_oauth_client_auth_handler(self.CLIENT_AUTH_BASIC)
    auth_handler.apply_client_authentication_options(headers, request_body, bearer_token)
    assert headers == {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(bearer_token)}
    assert request_body == {'foo': 'bar'}