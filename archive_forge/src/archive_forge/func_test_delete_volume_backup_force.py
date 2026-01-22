from openstack.block_storage.v3 import backup
from openstack.tests.unit import base
def test_delete_volume_backup_force(self):
    backup_id = '6ff16bdf-44d5-4bf9-b0f3-687549c76414'
    backup = {'id': backup_id, 'status': 'available'}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', 'detail']), json={'backups': [backup]}), dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['backups', backup_id, 'action']), json={'os-force_delete': None}, validate=dict(json={'os-force_delete': None})), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', backup_id]), json={'backup': backup}), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', backup_id]), status_code=404)])
    self.cloud.delete_volume_backup(backup_id, True, True, 1)
    self.assert_calls()