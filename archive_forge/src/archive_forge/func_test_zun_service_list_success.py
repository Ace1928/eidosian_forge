from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.availability_zones.AvailabilityZoneManager.list')
def test_zun_service_list_success(self, mock_list):
    self._test_arg_success('availability-zone-list')
    self.assertTrue(mock_list.called)