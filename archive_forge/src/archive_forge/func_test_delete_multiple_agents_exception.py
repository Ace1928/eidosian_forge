from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple_agents_exception(self):
    arglist = [self.fake_agents[0].agent_id, self.fake_agents[1].agent_id, 'x-y-z']
    verifylist = [('id', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ret_delete = [None, None, exceptions.NotFound('404')]
    self.agents_mock.delete = mock.Mock(side_effect=ret_delete)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    calls = [call(self.fake_agents[0].agent_id), call(self.fake_agents[1].agent_id)]
    self.agents_mock.delete.assert_has_calls(calls)