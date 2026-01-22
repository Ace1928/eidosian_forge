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
def test_apply_workforce_without_quota_project_id(self):
    headers = {}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE)
    credentials = self.make_workforce_pool_credentials(workforce_pool_user_project=self.WORKFORCE_POOL_USER_PROJECT)
    credentials.refresh(request)
    credentials.apply(headers)
    assert headers == {'authorization': 'Bearer {}'.format(self.SUCCESS_RESPONSE['access_token'])}