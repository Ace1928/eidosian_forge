from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
@mock.patch('magnumclient.v1.clusters.ClusterManager.create')
def test_cluster_create_deprecation_errors(self, mock_create, mock_get):
    self._test_arg_failure('cluster-create --cluster-template xxx --keypair-id x --keypair x', self._too_many_group_arg_error)
    self.assertFalse(mock_create.called)
    self.assertFalse(mock_get.called)