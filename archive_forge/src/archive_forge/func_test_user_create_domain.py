import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_create_domain(self):
    arglist = ['--domain', self.domain.name, self.user.name]
    verifylist = [('domain', self.domain.name), ('enable', False), ('disable', False), ('name', self.user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.user.name, 'default_project': None, 'description': None, 'domain': self.domain.id, 'email': None, 'options': {}, 'enabled': True, 'password': None}
    self.users_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)