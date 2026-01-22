import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import datastores
def test_get_by_tenant(self):
    page_mock = mock.Mock()
    self.datastore_version_members._list = page_mock
    limit = 'test-limit'
    marker = 'test-marker'
    self.datastore_version_members.get_by_tenant('datastore1', 'tenant1', limit, marker)
    page_mock.assert_called_with('/mgmt/datastores/datastore1/versions/members/tenant1', 'datastore_version_members', limit, marker)