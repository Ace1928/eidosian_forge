import socket
from unittest import mock
from octavia_lib.api.drivers import driver_lib
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
@mock.patch('octavia_lib.api.drivers.driver_lib.DriverLibrary._check_for_socket_ready.retry.sleep')
@mock.patch('os.path.exists')
def test_check_for_socket_ready(self, mock_path_exists, mock_sleep):
    mock_path_exists.return_value = True
    self.driver_lib._check_for_socket_ready('bogus')
    mock_path_exists.return_value = False
    self.assertRaises(driver_exceptions.DriverAgentNotFound, self.driver_lib._check_for_socket_ready, 'bogus')