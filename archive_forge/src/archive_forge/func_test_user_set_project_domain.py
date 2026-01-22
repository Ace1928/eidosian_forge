import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_set_project_domain(self):
    arglist = ['--project', self.project.id, '--project-domain', self.project.domain_id, self.user.name]
    verifylist = [('name', None), ('password', None), ('email', None), ('project', self.project.id), ('project_domain', self.project.domain_id), ('enable', False), ('disable', False), ('user', self.user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': True, 'default_project': self.project.id}
    self.users_mock.update.assert_called_with(self.user.id, **kwargs)
    self.assertIsNone(result)