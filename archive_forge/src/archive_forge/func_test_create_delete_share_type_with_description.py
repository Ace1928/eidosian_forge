import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
def test_create_delete_share_type_with_description(self):
    self.skip_if_microversion_not_supported('2.41')
    self._test_create_delete_share_type('2.41', True, False, None, None, None, None, None, description=data_utils.rand_name('test_share_type_description'))