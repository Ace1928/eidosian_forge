from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_failure_duplicate_name(self, mock_create):
    self._test_arg_failure('cluster-template-create foo --name test', self._mandatory_arg_error)
    mock_create.assert_not_called()