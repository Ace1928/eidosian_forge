from unittest import mock
import fixtures
from urllib import parse as urlparse
import uuid
from testtools import matchers
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient import utils as client_utils
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import auth
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
def test_update_consumers(self):
    body = {'consumer': {'name': 'admin'}}
    patch_mock = self._mock_request_method(method='patch', body=body)
    self._mock_request_method(method='post', body=body)
    response = self.mgr.update('admin', 'demo')
    self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
    patch_mock.assert_called_once_with('/OS-OAUTH1/consumers/admin', body={'consumer': {'description': 'demo'}})