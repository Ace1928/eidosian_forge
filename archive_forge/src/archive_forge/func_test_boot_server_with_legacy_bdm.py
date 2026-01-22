import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
def test_boot_server_with_legacy_bdm(self):
    params = ('', '', '1')
    self._boot_server_with_legacy_bdm(bdm_params=params)