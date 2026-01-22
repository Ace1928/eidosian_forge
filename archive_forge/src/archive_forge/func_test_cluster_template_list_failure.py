from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.list')
def test_cluster_template_list_failure(self, mock_list):
    self._test_arg_failure('cluster-template-list --wrong', self._unrecognized_arg_error)
    mock_list.assert_not_called()