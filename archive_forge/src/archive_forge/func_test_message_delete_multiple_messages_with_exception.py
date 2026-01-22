from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
def test_message_delete_multiple_messages_with_exception(self):
    self.volume_client.api_version = api_versions.APIVersion('3.3')
    arglist = [self.fake_messages[0].id, 'invalid_message']
    verifylist = [('message_ids', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.volume_messages_mock.delete.side_effect = [self.fake_messages[0], exceptions.CommandError]
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertEqual('Failed to delete 1 of 2 messages.', str(exc))
    self.volume_messages_mock.delete.assert_any_call(self.fake_messages[0].id)
    self.volume_messages_mock.delete.assert_any_call('invalid_message')
    self.assertEqual(2, self.volume_messages_mock.delete.call_count)