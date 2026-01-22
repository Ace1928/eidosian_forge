from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.mservices.MServiceManager.list')
def test_magnum_service_list_failure(self, mock_list):
    self._test_arg_failure('service-list --wrong', self._unrecognized_arg_error)
    mock_list.assert_not_called()