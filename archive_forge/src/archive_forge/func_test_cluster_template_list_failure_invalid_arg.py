from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.list')
def test_cluster_template_list_failure_invalid_arg(self, mock_list):
    _error_msg = ['.*?^usage: magnum cluster-template-list ', '.*?^error: argument --sort-dir: invalid choice: ', ".*?^Try 'magnum help cluster-template-list' for more information."]
    self._test_arg_failure('cluster-template-list --sort-dir aaa', _error_msg)
    mock_list.assert_not_called()