import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data('name', 'description')
def test_list_share_networks_filter_by_inexact(self, option):
    self.create_share_network(name=data_utils.rand_name('autotest_inexact'), description='fake_description_inexact', neutron_net_id='fake_neutron_net_id', neutron_subnet_id='fake_neutron_subnet_id')
    filters = {option + '~': 'inexact'}
    share_networks = self.admin_client.list_share_networks(filters=filters)
    self.assertGreater(len(share_networks), 0)