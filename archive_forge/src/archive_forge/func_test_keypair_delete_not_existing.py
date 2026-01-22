import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_keypair_delete_not_existing(self):
    """Try to delete keypair with not existing name.

        Test steps:
        1) Create keypair in setUp
        2) Try to delete not existing keypair
        """
    self.assertRaises(exceptions.CommandFailed, self.openstack, 'keypair delete not_existing')