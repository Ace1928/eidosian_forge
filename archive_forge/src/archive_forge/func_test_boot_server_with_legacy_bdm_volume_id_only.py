import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
def test_boot_server_with_legacy_bdm_volume_id_only(self):
    self._boot_server_with_legacy_bdm()