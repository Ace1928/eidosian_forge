from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_set_description(self):
    arglist = ['--description', self.fake_project.description, self.fake_project.name]
    verifylist = [('description', self.fake_project.description), ('enable', False), ('disable', False), ('project', self.fake_project.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'description': self.fake_project.description, 'enabled': True, 'tenant_name': self.fake_project.name}
    self.projects_mock.update.assert_called_with(self.fake_project.id, **kwargs)
    self.assertIsNone(result)