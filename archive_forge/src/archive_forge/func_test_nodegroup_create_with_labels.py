import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_create_with_labels(self):
    """Verifies labels are properly parsed when given as argument."""
    expected_args = self._default_args
    expected_args['labels'] = {'arg1': 'value1', 'arg2': 'value2'}
    arglist = ['--labels', 'arg1=value1', '--labels', 'arg2=value2', self.nodegroup.cluster_id, self.nodegroup.name]
    verifylist = [('labels', ['arg1=value1', 'arg2=value2']), ('name', self.nodegroup.name), ('cluster', self.nodegroup.cluster_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ng_mock.create.assert_called_with(self.nodegroup.cluster_id, **expected_args)