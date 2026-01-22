import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_list_shares_in_recycle_bin(self):
    shares = self.user_client.list_shares(is_soft_deleted=True)
    self.assertTrue(any((self.wait_soft_delete_share['id'] == s['ID'] for s in shares)))