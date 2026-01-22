from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.delete')
def test_quotas_delete_failure_no_resource(self, mock_delete):
    self._test_arg_failure('quotas-delete --project-id xxx', self._mandatory_arg_error)
    mock_delete.assert_not_called()