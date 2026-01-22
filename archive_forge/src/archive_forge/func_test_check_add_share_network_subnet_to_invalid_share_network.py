import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_check_add_share_network_subnet_to_invalid_share_network(self):
    self.assertRaises(tempest_lib_exc.CommandFailed, self.user_client.share_network_subnet_create_check, 'invalid_share_network', self.neutron_net_id, self.neutron_subnet_id)