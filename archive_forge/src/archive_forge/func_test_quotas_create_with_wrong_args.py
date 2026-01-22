from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_quotas_create_with_wrong_args(self):
    arglist = ['--project-id', 'abc', '--resources', 'Cluster', '--hard-limit', '10']
    verifylist = []
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)