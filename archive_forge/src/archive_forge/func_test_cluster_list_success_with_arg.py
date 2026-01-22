from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.list')
def test_cluster_list_success_with_arg(self, mock_list):
    self._test_arg_success('cluster-list --marker some_uuid --limit 1 --sort-dir asc --sort-key uuid')
    expected_args = self._get_expected_args_list('some_uuid', 1, 'asc', 'uuid')
    mock_list.assert_called_once_with(**expected_args)