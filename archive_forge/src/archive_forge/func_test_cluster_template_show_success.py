from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
def test_cluster_template_show_success(self, mock_show):
    self._test_arg_success('cluster-template-show xxx')
    mock_show.assert_called_once_with('xxx')