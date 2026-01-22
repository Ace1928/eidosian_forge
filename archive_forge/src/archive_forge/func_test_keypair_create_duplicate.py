import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_keypair_create_duplicate(self):
    """Try to create duplicate name keypair.

        Test steps:
        1) Create keypair in setUp
        2) Try to create duplicate keypair with the same name
        """
    self.assertRaises(exceptions.CommandFailed, self.openstack, 'keypair create ' + self.KPName)