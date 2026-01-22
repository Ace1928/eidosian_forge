from unittest import mock
from openstackclient.network.v2 import network_auto_allocated_topology
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_create_project_domain_option(self):
    arglist = ['--project', self.project.id, '--project-domain', self.project.domain_id]
    verifylist = [('project', self.project.id), ('project_domain', self.project.domain_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.get_auto_allocated_topology.assert_called_with(self.project.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)