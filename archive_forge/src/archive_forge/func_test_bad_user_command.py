from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional.identity.v2 import common
def test_bad_user_command(self):
    self.assertRaises(exceptions.CommandFailed, self.openstack, 'user unlist')