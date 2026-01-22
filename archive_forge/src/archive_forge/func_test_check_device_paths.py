from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.windows import base as base_win_conn
from os_brick.tests.windows import fake_win_conn
from os_brick.tests.windows import test_base
def test_check_device_paths(self):
    device_paths = [mock.sentinel.dev_path_0, mock.sentinel.dev_path_1]
    self.assertRaises(exception.BrickException, self._connector._check_device_paths, device_paths)