from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.delete')
def test_quotas_delete_success(self, mock_delete):
    self._test_arg_success('quotas-delete --project-id xxx --resource Cluster')
    mock_delete.assert_called_once_with('xxx', 'Cluster')