from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional.identity.v3 import common
def test_domain_delete_failure(self):
    domain_name = self._create_dummy_domain()
    self.assertRaises(exceptions.CommandFailed, self.openstack, 'domain delete %s' % domain_name)