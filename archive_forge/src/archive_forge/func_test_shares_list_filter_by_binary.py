import ddt
from manilaclient.tests.functional import base
def test_shares_list_filter_by_binary(self):
    self.clients['admin'].manila('service-list', params='--binary binary')