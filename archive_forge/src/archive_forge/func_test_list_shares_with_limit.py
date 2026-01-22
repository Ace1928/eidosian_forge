import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data({'limit': 1}, {'limit': 2}, {'limit': 1, 'offset': 1}, {'limit': 2, 'offset': 0})
def test_list_shares_with_limit(self, filters):
    shares = self.user_client.list_shares(filters=filters)
    self.assertEqual(filters['limit'], len(shares))