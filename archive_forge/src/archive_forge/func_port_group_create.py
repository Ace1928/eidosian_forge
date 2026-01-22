import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def port_group_create(self, node_id, name=None, params=''):
    """Create baremetal port group.

        :param String node_id: baremetal node UUID
        :param String name: port group name
        :param String params: Additional args and kwargs
        :return: JSON object of created port group
        """
    if not name:
        name = data_utils.rand_name('port_group')
    opts = self.get_opts()
    output = self.openstack('baremetal port group create {0} --node {1} --name {2} {3}'.format(opts, node_id, name, params))
    port_group = json.loads(output)
    if not port_group:
        self.fail('Baremetal port group has not been created!')
    self.addCleanup(self.port_group_delete, port_group['uuid'], params=params, ignore_exceptions=True)
    return port_group