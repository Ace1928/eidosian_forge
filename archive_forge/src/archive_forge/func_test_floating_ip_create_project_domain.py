from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_floating_ip_create_project_domain(self):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    self.projects_mock.get.return_value = project
    arglist = ['--project', project.name, '--project-domain', domain.name, self.floating_ip.floating_network_id]
    verifylist = [('network', self.floating_ip.floating_network_id), ('project', project.name), ('project_domain', domain.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_ip.assert_called_once_with(**{'floating_network_id': self.floating_ip.floating_network_id, 'project_id': project.id})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)