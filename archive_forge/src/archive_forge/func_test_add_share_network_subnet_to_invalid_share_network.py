import ddt
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
def test_add_share_network_subnet_to_invalid_share_network(self):
    self.assertRaises(exceptions.CommandFailed, self.add_share_network_subnet, 'invalid_share_network', self.neutron_net_id, self.neutron_subnet_id)