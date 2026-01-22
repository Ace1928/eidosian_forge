import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
@ddt.data('2.6', '2.7')
def test_list_share_type(self, microversion):
    share_type_name = data_utils.rand_name('manilaclient_functional_test')
    self.create_share_type(name=share_type_name, driver_handles_share_servers='False')
    share_types = self.admin_client.list_share_types(list_all=True, microversion=microversion)
    self.assertTrue(any((s['ID'] is not None for s in share_types)))
    self.assertTrue(any((s['Name'] is not None for s in share_types)))
    self.assertTrue(any((s['visibility'] is not None for s in share_types)))