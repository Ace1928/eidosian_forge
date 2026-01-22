import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_create_terminate_volume_image(self):
    self.skipTest('Volume functional tests temporarily disabled')
    if not self.user_cloud.has_service('volume'):
        self.skipTest('volume service not supported by cloud')
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    server = self.user_cloud.create_server(name=self.server_name, image=self.image, flavor=self.flavor, boot_from_volume=True, terminate_volume=True, volume_size=1, wait=True)
    volume_id = self._assert_volume_attach(server)
    self.assertTrue(self.user_cloud.delete_server(self.server_name, wait=True))
    volume = self.user_cloud.get_volume(volume_id)
    if volume:
        self.assertEqual('deleting', volume.status)
    srv = self.user_cloud.get_server(self.server_name)
    self.assertTrue(srv is None or srv.status.lower() == 'deleted')