from openstack.block_storage.v3 import backup
from openstack.tests.unit import base
def test_create_volume_backup(self):
    volume_id = '1234'
    backup_name = 'bak1'
    bak1 = {'id': '5678', 'volume_id': volume_id, 'status': 'available', 'name': backup_name}
    self.register_uris([dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['backups']), json={'backup': bak1}, validate=dict(json={'backup': {'name': backup_name, 'volume_id': volume_id, 'description': None, 'force': False, 'snapshot_id': None, 'incremental': False}}))])
    self.cloud.create_volume_backup(volume_id, name=backup_name)
    self.assert_calls()