import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_create_project_domain(self):
    arglist = ['--project', self.project.name, '--project-domain', self.project.domain_id, self.user.name]
    verifylist = [('project', self.project.name), ('project_domain', self.project.domain_id), ('enable', False), ('disable', False), ('name', self.user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.user.name, 'default_project': self.project.id, 'description': None, 'domain': None, 'email': None, 'options': {}, 'enabled': True, 'password': None}
    self.users_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    datalist = (self.project.id, self.domain.id, self.user.email, True, self.user.id, self.user.name)
    self.assertEqual(datalist, data)