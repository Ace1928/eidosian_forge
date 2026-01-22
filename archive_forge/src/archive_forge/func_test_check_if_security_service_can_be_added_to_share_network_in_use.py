import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_check_if_security_service_can_be_added_to_share_network_in_use(self):
    share_network = self.create_share_network(client=self.user_client, description='fakedescription', neutron_net_id='fake_neutron_net_id', neutron_subnet_id='fake_neutron_subnet_id')
    self.create_share(self.protocol, client=self.user_client, share_network=share_network['id'])
    current_security_service = self.create_security_service(client=self.user_client)
    check_result = self.user_client.share_network_security_service_add_check(share_network['id'], current_security_service['id'])
    self.assertEqual(check_result['compatible'], 'None')
    self._wait_for_update_security_service_compatible_result(share_network, current_security_service)