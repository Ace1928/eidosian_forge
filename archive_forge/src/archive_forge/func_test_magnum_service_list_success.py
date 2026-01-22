from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.mservices.MServiceManager.list')
def test_magnum_service_list_success(self, mock_list):
    self._test_arg_success('service-list')
    mock_list.assert_called_once_with()