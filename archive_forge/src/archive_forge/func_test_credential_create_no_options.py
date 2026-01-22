from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_credential_create_no_options(self):
    arglist = [self.credential.user_id, self.credential.blob]
    verifylist = [('user', self.credential.user_id), ('data', self.credential.blob)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'user': self.credential.user_id, 'type': self.credential.type, 'blob': self.credential.blob, 'project': None}
    self.credentials_mock.create.assert_called_once_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)