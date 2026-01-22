from unittest import mock
from magnumclient.common import cliutils
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1 import certificates_shell
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
@mock.patch('magnumclient.v1.certificates.CertificateManager.rotate_ca')
def test_ca_rotate(self, mock_rotate_ca, mock_cluster_get):
    mockcluster = mock.MagicMock()
    mockcluster.status = 'CREATE_COMPLETE'
    mockcluster.uuid = 'xxx'
    mock_cluster_get.return_value = mockcluster
    mock_rotate_ca.return_value = None
    self._test_arg_success('ca-rotate --cluster xxx')
    expected_args = {}
    expected_args['cluster_uuid'] = mockcluster.uuid
    mock_rotate_ca.assert_called_once_with(**expected_args)