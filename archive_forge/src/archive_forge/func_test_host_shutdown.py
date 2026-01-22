from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_host_shutdown(self):
    self._test_host_power_action(constants.HOST_POWER_ACTION_SHUTDOWN)