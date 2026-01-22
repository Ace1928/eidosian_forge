import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_snapshot(self):
    """Tests volume create from snapshot"""
    volume_name = uuid.uuid4().hex
    snapshot_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 1 ' + volume_name, parse_output=True)
    self.wait_for_status('volume', volume_name, 'available')
    self.assertEqual(volume_name, cmd_output['name'])
    cmd_output = self.openstack('volume snapshot create ' + snapshot_name + ' --volume ' + volume_name, parse_output=True)
    self.wait_for_status('volume snapshot', snapshot_name, 'available')
    name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--snapshot ' + snapshot_name + ' ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'volume delete ' + name)
    self.addCleanup(self.openstack, 'volume delete ' + volume_name)
    self.assertEqual(name, cmd_output['name'])
    self.wait_for_status('volume', name, 'available')
    raw_output = self.openstack('volume snapshot delete ' + snapshot_name)
    self.assertOutput('', raw_output)
    self.wait_for_delete('volume snapshot', snapshot_name)