import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_boot_server_with_no_network(self):
    """Tests that '--nic none' is honored.
        """
    server_info = self.nova('boot', params='%(name)s --flavor %(flavor)s --poll --image %(image)s --nic none' % {'name': self.name_generate(), 'flavor': self.flavor.id, 'image': self.image.id})
    server_id = self._get_value_from_the_table(server_info, 'id')
    self.addCleanup(self.wait_for_resource_delete, server_id, self.client.servers)
    self.addCleanup(self.client.servers.delete, server_id)
    server_info = self.nova('show', params=server_id)
    network = self._find_network_in_table(server_info)
    self.assertIsNone(network, 'Unexpected network allocation: %s' % server_info)