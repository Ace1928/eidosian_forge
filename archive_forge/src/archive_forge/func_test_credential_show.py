from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_credential_show(self):
    arglist = [self.credential.id]
    verifylist = [('credential', self.credential.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.credentials_mock.get.assert_called_once_with(self.credential.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)