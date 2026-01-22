import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
def test_boot_server_with_net_name(self):
    server_info = self.nova('boot', params='%(name)s --flavor %(flavor)s --image %(image)s --poll --nic net-name=%(net-name)s' % {'name': self.name_generate(), 'image': self.image.id, 'flavor': self.flavor.id, 'net-name': self.network.name})
    server_id = self._get_value_from_the_table(server_info, 'id')
    self.client.servers.delete(server_id)
    self.wait_for_resource_delete(server_id, self.client.servers)