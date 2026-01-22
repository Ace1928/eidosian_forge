import uuid
import fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
import requests
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
from keystoneclient import utils as base_utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import roles
from keystoneclient.v3 import users
def test_list_with_multiple_response_objects(self):
    body = {'hello': [{'name': 'admin'}, {'name': 'admin'}]}
    resp_1 = requests.Response()
    resp_1.headers['x-openstack-request-id'] = TEST_REQUEST_ID
    resp_2 = requests.Response()
    resp_2.headers['x-openstack-request-id'] = TEST_REQUEST_ID_1
    resp_result = [resp_1, resp_2]
    get_mock = self.useFixture(fixtures.MockPatchObject(self.client, 'get', autospec=True, return_value=(resp_result, body))).mock
    returned_list = self.mgr._list(self.url, 'hello')
    self.assertIn(returned_list.request_ids[0], [TEST_REQUEST_ID, TEST_REQUEST_ID_1])
    self.assertIn(returned_list.request_ids[1], [TEST_REQUEST_ID, TEST_REQUEST_ID_1])
    get_mock.assert_called_once_with(self.url)