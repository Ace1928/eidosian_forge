from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
@mock.patch('magnumclient.v1.clusters.ClusterManager.create')
def test_cluster_create_success(self, mock_create, mock_get):
    mock_ct = mock.MagicMock()
    mock_ct.uuid = 'xxx'
    mock_get.return_value = mock_ct
    self._test_arg_success('cluster-create test --cluster-template xxx --node-count 123 --timeout 15')
    expected_args = self._get_expected_args_create('xxx', name='test', node_count=123, create_timeout=15)
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-create --cluster-template xxx')
    expected_args = self._get_expected_args_create('xxx')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-create --cluster-template xxx --keypair x')
    expected_args = self._get_expected_args_create('xxx', keypair='x')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-create --cluster-template xxx --docker-volume-size 20')
    expected_args = self._get_expected_args_create('xxx', docker_volume_size=20)
    self._test_arg_success('cluster-create --cluster-template xxx --labels key=val')
    expected_args = self._get_expected_args_create('xxx', labels={'key': 'val'})
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-create test --cluster-template xxx')
    expected_args = self._get_expected_args_create('xxx', name='test')
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-create --cluster-template xxx --node-count 123')
    expected_args = self._get_expected_args_create('xxx', node_count=123)
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-create --cluster-template xxx --node-count 123 --master-count 123')
    expected_args = self._get_expected_args_create('xxx', master_count=123, node_count=123)
    mock_create.assert_called_with(**expected_args)
    self._test_arg_success('cluster-create --cluster-template xxx --timeout 15')
    expected_args = self._get_expected_args_create('xxx', create_timeout=15)
    mock_create.assert_called_with(**expected_args)