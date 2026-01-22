from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.tests.functional import base
def test_pools_list(self):
    self.clients['admin'].manila('pool-list')