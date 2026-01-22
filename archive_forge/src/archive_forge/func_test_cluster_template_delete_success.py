from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.delete')
def test_cluster_template_delete_success(self, mock_delete):
    self._test_arg_success('cluster-template-delete xxx')
    mock_delete.assert_called_once_with('xxx')