from openstack.block_storage.v3 import backup
from openstack.tests.unit import base
def test_delete_volume_backup_wait(self):
    backup_id = '6ff16bdf-44d5-4bf9-b0f3-687549c76414'
    backup = {'id': backup_id, 'status': 'available'}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', 'detail']), json={'backups': [backup]}), dict(method='DELETE', uri=self.get_mock_url('volumev3', 'public', append=['backups', backup_id])), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', backup_id]), json={'backup': backup}), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', backup_id]), status_code=404)])
    self.cloud.delete_volume_backup(backup_id, False, True, 1)
    self.assert_calls()