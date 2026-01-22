from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_agent_delete(self):
    arglist = [self.network_agents[0].id]
    verifylist = [('network_agent', [self.network_agents[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.delete_agent.assert_called_once_with(self.network_agents[0].id, ignore_missing=False)
    self.assertIsNone(result)