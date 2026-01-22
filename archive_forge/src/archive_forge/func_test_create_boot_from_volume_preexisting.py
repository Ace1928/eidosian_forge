import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_create_boot_from_volume_preexisting(self):
    self.skipTest('Volume functional tests temporarily disabled')
    if not self.user_cloud.has_service('volume'):
        self.skipTest('volume service not supported by cloud')
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    volume = self.user_cloud.create_volume(size=1, name=self.server_name, image=self.image, wait=True)
    self.addCleanup(self.user_cloud.delete_volume, volume.id)
    server = self.user_cloud.create_server(name=self.server_name, image=None, flavor=self.flavor, boot_volume=volume, volume_size=1, wait=True)
    volume_id = self._assert_volume_attach(server, volume_id=volume['id'])
    self.assertTrue(self.user_cloud.delete_server(self.server_name, wait=True))
    volume = self.user_cloud.get_volume(volume_id)
    self.assertIsNotNone(volume)
    self.assertEqual(volume['name'], volume['display_name'])
    self.assertTrue(volume['bootable'])
    self.assertEqual([], volume['attachments'])
    self._wait_for_detach(volume.id)
    self.assertTrue(self.user_cloud.delete_volume(volume_id))
    srv = self.user_cloud.get_server(self.server_name)
    self.assertTrue(srv is None or srv.status.lower() == 'deleted')
    self.assertIsNone(self.user_cloud.get_volume(volume_id))