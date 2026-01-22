import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_share_networks_select_column(self):
    share_networks = self.admin_client.list_share_networks(columns='id')
    self.assertTrue(any((s['Id'] is not None for s in share_networks)))
    self.assertTrue(all(('Name' not in s for s in share_networks)))
    self.assertTrue(all(('name' not in s for s in share_networks)))