from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_one_agent(self):
    arglist = [self.fake_agents[0].agent_id]
    verifylist = [('id', [self.fake_agents[0].agent_id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.agents_mock.delete.assert_called_with(self.fake_agents[0].agent_id)
    self.assertIsNone(result)