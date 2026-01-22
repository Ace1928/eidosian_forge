import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_delete_with_domain(self):
    self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
    self.roles_mock.delete.return_value = None
    arglist = ['--domain', identity_fakes.domain_name, identity_fakes.ROLE_2['name']]
    verifylist = [('roles', [identity_fakes.ROLE_2['name']]), ('domain', identity_fakes.domain_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.roles_mock.delete.assert_called_with(identity_fakes.ROLE_2['id'])
    self.assertIsNone(result)