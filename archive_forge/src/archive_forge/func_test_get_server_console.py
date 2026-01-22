import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_get_server_console(self):
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    server = self.user_cloud.create_server(name=self.server_name, image=self.image, flavor=self.flavor, wait=True)
    log = self.user_cloud._get_server_console_output(server_id=server.id)
    self.assertIsInstance(log, str)