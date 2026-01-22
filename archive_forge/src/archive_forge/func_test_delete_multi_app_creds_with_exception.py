import copy
import json
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import application_credential
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
@mock.patch.object(utils, 'find_resource')
def test_delete_multi_app_creds_with_exception(self, find_mock):
    find_mock.side_effect = [self.app_creds_mock.get.return_value, exceptions.CommandError]
    arglist = [identity_fakes.app_cred_id, 'nonexistent_app_cred']
    verifylist = [('application_credential', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 application credentials failed to delete.', str(e))
    find_mock.assert_any_call(self.app_creds_mock, identity_fakes.app_cred_id)
    find_mock.assert_any_call(self.app_creds_mock, 'nonexistent_app_cred')
    self.assertEqual(2, find_mock.call_count)
    self.app_creds_mock.delete.assert_called_once_with(identity_fakes.app_cred_id)