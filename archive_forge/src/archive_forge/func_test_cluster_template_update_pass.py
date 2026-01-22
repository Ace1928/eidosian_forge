import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_update_pass(self):
    arglist = ['foo', 'remove', 'bar']
    verifylist = [('cluster-template', 'foo'), ('op', 'remove'), ('attributes', [['bar']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.cluster_templates_mock.update.assert_called_with('foo', [{'op': 'remove', 'path': '/bar'}])