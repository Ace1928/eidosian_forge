import uuid
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cleanup
def test_block_storage_cleanup_pre_324(self):
    arglist = []
    verifylist = [('cluster', None), ('host', None), ('binary', None), ('is_up', None), ('disabled', None), ('resource_id', None), ('resource_type', None), ('service_id', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.24 or greater is required', str(exc))