from unittest import mock
from magnumclient.common import cliutils
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1 import certificates_shell
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
@mock.patch('magnumclient.v1.certificates.CertificateManager.get')
def test_ca_show_failure_duplicate_arg(self, mock_cert_get, mock_cluster_get):
    self.assertRaises(cliutils.DuplicateArgs, self._test_arg_failure, 'ca-show foo --cluster foo', self._duplicate_arg_error)
    mock_cert_get.assert_not_called()
    mock_cluster_get.assert_not_called()