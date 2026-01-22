from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import quotas
@ddt.data('2.39', REPLICA_QUOTAS_MICROVERSION)
def test_update_share_type_quota(self, microversion):
    tenant_id = 'fake_tenant_id'
    share_type = 'fake_share_type'
    manager = self._get_manager(microversion)
    resource_path = self._get_resource_path(microversion)
    expected_url = '%s/%s?share_type=%s' % (resource_path, tenant_id, share_type)
    expected_body = {'quota_set': {'tenant_id': tenant_id, 'shares': 1, 'snapshots': 2, 'gigabytes': 3, 'snapshot_gigabytes': 4}}
    kwargs = {}
    if microversion >= REPLICA_QUOTAS_MICROVERSION:
        expected_body['quota_set']['share_replicas'] = 8
        expected_body['quota_set']['replica_gigabytes'] = 9
        kwargs = {'share_replicas': 8, 'replica_gigabytes': 9}
    if microversion >= '2.62':
        expected_body['quota_set']['per_share_gigabytes'] = 10
        kwargs = {'per_share_gigabytes': 10}
    with mock.patch.object(manager, '_update', mock.Mock(return_value='fake_update')):
        manager.update(tenant_id, shares=1, snapshots=2, gigabytes=3, snapshot_gigabytes=4, share_type=share_type, **kwargs)
        manager._update.assert_called_once_with(expected_url, expected_body, 'quota_set')