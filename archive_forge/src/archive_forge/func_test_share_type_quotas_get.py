from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import quotas
def test_share_type_quotas_get(self):
    tenant_id = 'fake_tenant_id'
    share_type = 'fake_share_type'
    manager = self._get_manager('2.39')
    resource_path = self._get_resource_path('2.39')
    expected_url = '%s/%s/detail?share_type=%s' % (resource_path, tenant_id, share_type)
    with mock.patch.object(manager, '_get', mock.Mock(return_value='fake_get')):
        manager.get(tenant_id, share_type=share_type, detail=True)
        manager._get.assert_called_once_with(expected_url, 'quota_set')