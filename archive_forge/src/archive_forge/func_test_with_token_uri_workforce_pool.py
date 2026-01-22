import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
def test_with_token_uri_workforce_pool(self):
    credentials = self.make_workforce_pool_credentials(workforce_pool_user_project=self.WORKFORCE_POOL_USER_PROJECT)
    new_token_uri = 'https://eu-sts.googleapis.com/v1/token'
    assert credentials._token_url == self.TOKEN_URL
    creds_with_new_token_uri = credentials.with_token_uri(new_token_uri)
    assert creds_with_new_token_uri._token_url == new_token_uri
    assert creds_with_new_token_uri.info.get('workforce_pool_user_project') == self.WORKFORCE_POOL_USER_PROJECT