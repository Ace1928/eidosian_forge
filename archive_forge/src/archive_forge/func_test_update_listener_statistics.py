import socket
from unittest import mock
from octavia_lib.api.drivers import driver_lib
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
@mock.patch('octavia_lib.api.drivers.driver_lib.DriverLibrary._send')
def test_update_listener_statistics(self, mock_send):
    error_dict = {'status_code': 500, 'fault_string': 'boom', 'status_object': 'balloon', 'status_object_id': '1', 'status_record': 'tunes'}
    mock_send.side_effect = [{'status_code': 200}, Exception('boom'), error_dict]
    self.driver_lib.update_listener_statistics('fake_stats')
    mock_send.assert_called_once_with('/var/run/octavia/stats.sock', 'fake_stats')
    self.assertRaises(driver_exceptions.UpdateStatisticsError, self.driver_lib.update_listener_statistics, 'fake_stats')
    self.assertRaises(driver_exceptions.UpdateStatisticsError, self.driver_lib.update_listener_statistics, 'fake_stats')