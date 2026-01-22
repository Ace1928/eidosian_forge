import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_registered_limit_delete(self):
    self.registered_limit_mock.delete.return_value = None
    arglist = [identity_fakes.registered_limit_id]
    verifylist = [('registered_limit_id', [identity_fakes.registered_limit_id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.registered_limit_mock.delete.assert_called_with(identity_fakes.registered_limit_id)
    self.assertIsNone(result)