from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.availability_zones.AvailabilityZoneManager.list')
def test_zun_service_list_failure(self, mock_list):
    self._test_arg_failure('availability-zone-list --wrong', self._unrecognized_arg_error)
    self.assertFalse(mock_list.called)