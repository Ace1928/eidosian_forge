from unittest import mock
from magnumclient.osc.v1 import stats as osc_stats
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_stats_list(self):
    arglist = ['abc']
    verifylist = [('project_id', 'abc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.clusters_mock.list.assert_called_once_with(project_id='abc')