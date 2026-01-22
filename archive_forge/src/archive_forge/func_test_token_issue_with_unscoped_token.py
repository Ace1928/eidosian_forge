from unittest import mock
from openstackclient.identity.v2_0 import token
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_token_issue_with_unscoped_token(self):
    auth_ref = identity_fakes.fake_auth_ref(identity_fakes.UNSCOPED_TOKEN)
    self.ar_mock = mock.PropertyMock(return_value=auth_ref)
    type(self.app.client_manager).auth_ref = self.ar_mock
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    collist = ('expires', 'id', 'user_id')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.token_expires, identity_fakes.token_id, 'user-id')
    self.assertEqual(datalist, data)