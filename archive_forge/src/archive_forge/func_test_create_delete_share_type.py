import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
@ddt.data(*unit_test_types.get_valid_type_create_data_2_0())
@ddt.unpack
def test_create_delete_share_type(self, is_public, dhss, spec_snapshot_support, extra_specs):
    self.skip_if_microversion_not_supported('2.0')
    self._test_create_delete_share_type('2.0', is_public, dhss, spec_snapshot_support, None, None, None, extra_specs)