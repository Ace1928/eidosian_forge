import ddt
from tempest.lib import exceptions
from cinderclient.tests.functional import base
def test_volume_create_from_snapshot(self):
    """Test steps:

        1) create volume in Setup()
        2) create snapshot
        3) create volume from snapshot
        4) check that volume from snapshot has been successfully created
        """
    snapshot = self.object_create('snapshot', params=self.volume['id'])
    volume_from_snapshot = self.object_create('volume', params='--snapshot-id {0} 1'.format(snapshot['id']))
    self.object_delete('snapshot', snapshot['id'])
    self.check_object_deleted('snapshot', snapshot['id'])
    cinder_list = self.cinder('list')
    self.assertIn(volume_from_snapshot['id'], cinder_list)