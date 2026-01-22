from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.update')
def test_cluster_update_rollback_old_api_version(self, mock_update):
    self.assertRaises(exceptions.CommandError, self.shell, '--magnum-api-version 1.2 cluster-update test add test=test --rollback')
    mock_update.assert_not_called()