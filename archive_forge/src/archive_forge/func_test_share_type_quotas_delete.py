from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import quotas
def test_share_type_quotas_delete(self):
    tenant_id = 'test'
    share_type = 'fake_st'
    manager = self._get_manager('2.39')
    resource_path = self._get_resource_path('2.39')
    expected_url = '%s/test?share_type=fake_st' % resource_path
    with mock.patch.object(manager, '_delete', mock.Mock(return_value='fake_delete')):
        manager.delete(tenant_id, share_type=share_type)
        manager._delete.assert_called_once_with(expected_url)