from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_show_ca(self):
    arglist = ['fake-cluster']
    verifylist = [('cluster', 'fake-cluster')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.clusters_mock.get.assert_called_once_with('fake-cluster')