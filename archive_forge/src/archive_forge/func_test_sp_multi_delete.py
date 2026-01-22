from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_sp_multi_delete(self):
    sp1 = self._create_dummy_sp(add_clean_up=False)
    sp2 = self._create_dummy_sp(add_clean_up=False)
    raw_output = self.openstack('service provider delete %s %s' % (sp1, sp2))
    self.assertEqual(0, len(raw_output))