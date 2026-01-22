from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_idp_multi_delete(self):
    idp_1 = self._create_dummy_idp(add_clean_up=False)
    idp_2 = self._create_dummy_idp(add_clean_up=False)
    raw_output = self.openstack('identity provider delete %s %s' % (idp_1, idp_2))
    self.assertEqual(0, len(raw_output))