import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_transfer_request_list_show(self):
    volume_name = uuid.uuid4().hex
    xfer_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 1 ' + volume_name, parse_output=True)
    self.assertEqual(volume_name, cmd_output['name'])
    self.addCleanup(self.openstack, '--os-volume-api-version ' + self.API_VERSION + ' ' + 'volume delete ' + volume_name)
    self.wait_for_status('volume', volume_name, 'available')
    cmd_output = self.openstack('--os-volume-api-version ' + self.API_VERSION + ' ' + 'volume transfer request create ' + ' --name ' + xfer_name + ' ' + volume_name, parse_output=True)
    self.assertEqual(xfer_name, cmd_output['name'])
    xfer_id = cmd_output['id']
    auth_key = cmd_output['auth_key']
    self.assertTrue(auth_key)
    self.wait_for_status('volume', volume_name, 'awaiting-transfer')
    cmd_output = self.openstack('--os-volume-api-version ' + self.API_VERSION + ' ' + 'volume transfer request list', parse_output=True)
    self.assertIn(xfer_name, [req['Name'] for req in cmd_output])
    cmd_output = self.openstack('--os-volume-api-version ' + self.API_VERSION + ' ' + 'volume transfer request show ' + xfer_id, parse_output=True)
    self.assertEqual(xfer_name, cmd_output['name'])
    cmd_output = self.openstack('--os-volume-api-version ' + self.API_VERSION + ' ' + 'volume transfer request delete ' + xfer_id)
    self.wait_for_status('volume', volume_name, 'available')