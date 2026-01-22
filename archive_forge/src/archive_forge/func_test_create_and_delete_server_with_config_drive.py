import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_create_and_delete_server_with_config_drive(self):
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    server = self.user_cloud.create_server(name=self.server_name, image=self.image, flavor=self.flavor, config_drive=True, wait=True)
    self.assertEqual(self.server_name, server['name'])
    self.assertEqual(self.image.id, server['image']['id'])
    self.assertEqual(self.flavor.name, server['flavor']['original_name'])
    self.assertTrue(server['has_config_drive'])
    self.assertIsNotNone(server['adminPass'])
    self.assertTrue(self.user_cloud.delete_server(self.server_name, wait=True))
    srv = self.user_cloud.get_server(self.server_name)
    self.assertTrue(srv is None or srv.status.lower() == 'deleted')