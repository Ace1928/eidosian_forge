import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_set_with_no_immutable(self):
    self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
    arglist = ['--name', 'over', '--no-immutable', identity_fakes.ROLE_2['name']]
    verifylist = [('name', 'over'), ('no_immutable', True), ('role', identity_fakes.ROLE_2['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'name': 'over', 'description': None, 'options': {'immutable': False}}
    self.roles_mock.update.assert_called_with(identity_fakes.ROLE_2['id'], **kwargs)
    self.assertIsNone(result)