from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
from openstackclient.api import object_store_v1 as object_store
from openstackclient.tests.unit import utils
def test_object_list_no_options(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/qaz', json=LIST_OBJECT_RESP, status_code=200)
    ret = self.api.object_list(container='qaz')
    self.assertEqual(LIST_OBJECT_RESP, ret)