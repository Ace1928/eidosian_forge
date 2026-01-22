from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_sp_delete(self):
    service_provider = self._create_dummy_sp(add_clean_up=False)
    raw_output = self.openstack('service provider delete %s' % service_provider)
    self.assertEqual(0, len(raw_output))