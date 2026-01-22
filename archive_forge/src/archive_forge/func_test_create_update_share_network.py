import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data(({'name': data_utils.rand_name('autotest_share_network_name')}, {}), ({'description': 'fake_description'}, {}), ({'neutron_net_id': 'fake_neutron_net_id', 'neutron_subnet_id': 'fake_neutron_subnet_id'}, {}), ({'name': '""'}, {}), ({'description': '""'}, {}), ({'neutron_net_id': '""'}, {'neutron_net_id': 'fake_nn_id', 'neutron_subnet_id': 'fake_nsn_id'}), ({'neutron_subnet_id': '""'}, {'neutron_net_id': 'fake_nn_id', 'neutron_subnet_id': 'fake_nsn_id'}))
@ddt.unpack
def test_create_update_share_network(self, net_data, net_creation_data):
    sn = self.create_share_network(cleanup_in_class=False, **net_creation_data)
    update = self.admin_client.update_share_network(sn['id'], **net_data)
    expected_nn_id, expected_nsn_id = self._get_expected_update_data(net_data, net_creation_data)
    expected_data = {'name': 'None', 'description': 'None', 'neutron_net_id': expected_nn_id, 'neutron_subnet_id': expected_nsn_id}
    subnet_keys = []
    if utils.share_network_subnets_are_supported():
        subnet_keys = ['neutron_net_id', 'neutron_subnet_id']
        subnet = ast.literal_eval(update['share_network_subnets'])
    update_values = dict([(k, v) for k, v in net_data.items() if v != '""'])
    expected_data.update(update_values)
    for k, v in expected_data.items():
        if k in subnet_keys:
            self.assertEqual(v, subnet[0][k])
        else:
            self.assertEqual(v, update[k])
    self.admin_client.delete_share_network(sn['id'])
    self.admin_client.wait_for_share_network_deletion(sn['id'])