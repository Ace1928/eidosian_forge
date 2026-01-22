from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.list')
def test_quotas_list_failure(self, mock_list):
    self._test_arg_failure('quotas-list --wrong', self._unrecognized_arg_error)
    mock_list.assert_not_called()