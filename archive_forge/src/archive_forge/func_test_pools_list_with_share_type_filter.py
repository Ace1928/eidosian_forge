from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.tests.functional import base
def test_pools_list_with_share_type_filter(self):
    share_type = self.create_share_type(name=data_utils.rand_name('manilaclient_functional_test'), snapshot_support=True)
    self.clients['admin'].manila('pool-list', params='--share_type ' + share_type['ID'])