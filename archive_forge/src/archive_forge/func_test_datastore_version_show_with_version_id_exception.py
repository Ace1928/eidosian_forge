from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
def test_datastore_version_show_with_version_id_exception(self):
    args = ['v-56']
    verifylist = [('datastore_version', 'v-56')]
    parsed_args = self.check_parser(self.cmd, args, verifylist)
    self.assertRaises(exceptions.NoUniqueMatch, self.cmd.take_action, parsed_args)