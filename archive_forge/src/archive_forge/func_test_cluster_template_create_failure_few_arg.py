from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.create')
def test_cluster_template_create_failure_few_arg(self, mock_create):
    self._test_arg_failure('cluster-template-create --name test', self._mandatory_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --image-id test', self._mandatory_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --keypair-id test', self._mandatory_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --external-network-id test', self._mandatory_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --coe test', self._mandatory_group_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --coe test --external-network test ', self._mandatory_group_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --coe test --image test ', self._mandatory_group_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create --server-type test', self._mandatory_arg_error)
    mock_create.assert_not_called()
    self._test_arg_failure('cluster-template-create', self._mandatory_arg_error)
    mock_create.assert_not_called()