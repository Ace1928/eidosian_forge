from unittest import mock
from magnumclient.common import cliutils
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1 import certificates_shell
@mock.patch('os.path.isfile')
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
@mock.patch('magnumclient.v1.certificates.CertificateManager.create')
def test_cluster_ca_sign_success(self, mock_cert_create, mock_cluster_get, mock_isfile):
    mock_isfile.return_value = True
    mockcluster = mock.MagicMock()
    mockcluster.status = 'CREATE_COMPLETE'
    mockcluster.uuid = 'xxx'
    mock_cluster_get.return_value = mockcluster
    fake_csr = 'fake-csr'
    mock_file = mock.mock_open(read_data=fake_csr)
    with mock.patch.object(certificates_shell, 'open', mock_file):
        self._test_arg_success('ca-sign --csr path/csr.pem --cluster xxx')
        expected_args = {}
        expected_args['cluster_uuid'] = mockcluster.uuid
        expected_args['csr'] = fake_csr
        mock_cert_create.assert_called_once_with(**expected_args)