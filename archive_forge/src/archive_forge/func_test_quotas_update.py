from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_quotas_update(self):
    arglist = ['--project-id', 'abc', '--resource', 'Cluster', '--hard-limit', '10']
    verifylist = [('project_id', 'abc'), ('resource', 'Cluster'), ('hard_limit', 10)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.quotas_mock.update.assert_called_with('abc', 'Cluster', self._default_args)