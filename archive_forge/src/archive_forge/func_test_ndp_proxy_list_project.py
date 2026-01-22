from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_ndp_proxy_list_project(self):
    project = identity_fakes_v3.FakeProject.create_one_project()
    self.projects_mock.get.return_value = project
    arglist = ['--project', project.id]
    verifylist = [('project', project.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.ndp_proxies.assert_called_once_with(**{'project_id': project.id})
    self.assertEqual(self.columns, columns)
    self.assertItemsEqual(self.data, list(data))