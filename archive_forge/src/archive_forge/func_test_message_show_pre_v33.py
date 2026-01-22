from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
def test_message_show_pre_v33(self):
    self.volume_client.api_version = api_versions.APIVersion('3.2')
    arglist = [self.fake_message.id]
    verifylist = [('message_id', self.fake_message.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.3 or greater is required', str(exc))