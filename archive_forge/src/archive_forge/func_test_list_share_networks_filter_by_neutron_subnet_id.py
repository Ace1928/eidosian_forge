import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_share_networks_filter_by_neutron_subnet_id(self):
    filters = {'neutron_subnet_id': self.neutron_subnet_id}
    self._list_share_networks_with_filters(filters)