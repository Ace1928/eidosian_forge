import copy
from keystoneauth1 import token_endpoint
from osc_lib.tests import utils as osc_lib_test_utils
from openstackclient.common import clientmanager
from openstackclient.tests.unit import fakes
def test_client_manager_network_endpoint_disabled(self):
    auth_args = copy.deepcopy(self.default_password_auth)
    auth_args.update({'user_domain_name': 'default', 'project_domain_name': 'default'})
    client_manager = self._make_clientmanager(auth_args=auth_args, identity_api_version='3', auth_plugin_name='v3password')
    self.assertFalse(client_manager.is_service_available('network'))
    self.assertTrue(client_manager.is_network_endpoint_enabled())