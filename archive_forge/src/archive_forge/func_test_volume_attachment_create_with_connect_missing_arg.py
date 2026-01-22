from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_attachment
def test_volume_attachment_create_with_connect_missing_arg(self):
    self.volume_client.api_version = api_versions.APIVersion('3.54')
    arglist = [self.volume.id, self.server.id, '--initiator', 'iqn.1993-08.org.debian:01:cad181614cec']
    verifylist = [('volume', self.volume.id), ('server', self.server.id), ('connect', False), ('initiator', 'iqn.1993-08.org.debian:01:cad181614cec')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('You must specify the --connect option for any', str(exc))