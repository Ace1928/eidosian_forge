import copy
import json
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import application_credential
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_application_credential_delete(self):
    arglist = [identity_fakes.app_cred_id]
    verifylist = [('application_credential', [identity_fakes.app_cred_id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.app_creds_mock.delete.assert_called_with(identity_fakes.app_cred_id)
    self.assertIsNone(result)