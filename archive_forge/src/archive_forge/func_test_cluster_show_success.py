from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
def test_cluster_show_success(self, mock_show):
    self._test_arg_success('cluster-show xxx')
    mock_show.assert_called_once_with('xxx')