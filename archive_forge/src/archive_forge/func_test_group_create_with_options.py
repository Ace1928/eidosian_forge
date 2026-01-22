from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_create_with_options(self):
    arglist = ['--domain', self.domain.name, '--description', self.group.description, self.group.name]
    verifylist = [('domain', self.domain.name), ('description', self.group.description), ('name', self.group.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.groups_mock.create.assert_called_once_with(name=self.group.name, domain=self.domain.id, description=self.group.description)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)