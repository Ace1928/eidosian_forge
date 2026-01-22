import ddt
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
@ddt.data({'neutron_net_id': None, 'neutron_subnet_id': 'fake_subnet_id'}, {'neutron_net_id': 'fake_net_id', 'neutron_subnet_id': None}, {'availability_zone': 'invalid_availability_zone'})
def test_add_invalid_share_network_subnet_to_share_network(self, params):
    self.assertRaises(exceptions.CommandFailed, self.add_share_network_subnet, self.sn['id'], **params)