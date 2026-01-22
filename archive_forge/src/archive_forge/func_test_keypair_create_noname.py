import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_keypair_create_noname(self):
    """Try to create keypair without name.

        Test steps:
        1) Try to create keypair without a name
        """
    self.assertRaises(exceptions.CommandFailed, self.openstack, 'keypair create')