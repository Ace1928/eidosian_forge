import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_add_and_update_security_service_when_share_network_is_in_use(self):
    share_network = self.create_share_network(client=self.user_client, name='cool_net_name', description='fakedescription', neutron_net_id='fake_neutron_net_id', neutron_subnet_id='fake_neutron_subnet_id')
    self.create_share(self.protocol, name='fake_share_name', share_network=share_network['id'], client=self.user_client)
    current_security_service = self.create_security_service(client=self.user_client, name='current_security_service')
    new_security_service = self.create_security_service(client=self.user_client, name='new_security_service')
    check_result = self.user_client.share_network_security_service_add_check(share_network['id'], current_security_service['id'])
    self.assertEqual(check_result['compatible'], 'None')
    self._wait_for_update_security_service_compatible_result(share_network, current_security_service)
    self.user_client.share_network_security_service_add(share_network['id'], current_security_service['id'])
    network_services = self.user_client.share_network_security_service_list(share_network['id'])
    self.assertEqual(len(network_services), 1)
    self.assertEqual(network_services[0]['name'], current_security_service['name'])
    self.user_client.wait_for_resource_status(share_network['id'], 'active', microversion=SECURITY_SERVICE_UPDATE_VERSION, resource_type='share_network')
    check_result = self.user_client.share_network_security_service_update_check(share_network['id'], current_security_service['id'], new_security_service['id'])
    self.assertEqual(check_result['compatible'], 'None')
    self._wait_for_update_security_service_compatible_result(share_network, current_security_service, new_security_service=new_security_service)
    self.user_client.share_network_security_service_update(share_network['id'], current_security_service['id'], new_security_service['id'])
    network_services = self.user_client.share_network_security_service_list(share_network['id'])
    self.assertEqual(len(network_services), 1)
    self.assertEqual(network_services[0]['name'], new_security_service['name'])
    self.user_client.wait_for_resource_status(share_network['id'], 'active', microversion=SECURITY_SERVICE_UPDATE_VERSION, resource_type='share_network')