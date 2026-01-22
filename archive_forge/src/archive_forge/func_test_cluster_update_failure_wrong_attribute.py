from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.update')
def test_cluster_update_failure_wrong_attribute(self, mock_update):
    _error_msg = ['.*?^ERROR: Attributes must be a list of PATH=VALUE']
    self.assertRaises(exceptions.CommandError, self._test_arg_failure, 'cluster-update test add test', _error_msg)
    mock_update.assert_not_called()