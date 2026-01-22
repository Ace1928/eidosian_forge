from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.list')
def test_cluster_list_failure_invalid_arg(self, mock_list):
    _error_msg = ['.*?^usage: magnum cluster-list ', '.*?^error: argument --sort-dir: invalid choice: ', ".*?^Try 'magnum help cluster-list' for more information."]
    self._test_arg_failure('cluster-list --sort-dir aaa', _error_msg)
    mock_list.assert_not_called()