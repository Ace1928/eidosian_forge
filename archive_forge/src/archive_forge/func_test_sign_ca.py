from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_sign_ca(self):
    arglist = ['fake-cluster', self.test_csr_path]
    verifylist = [('cluster', 'fake-cluster'), ('csr', self.test_csr_path)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.clusters_mock.get.assert_called_once_with('fake-cluster')