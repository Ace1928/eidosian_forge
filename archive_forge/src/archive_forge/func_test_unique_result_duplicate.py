from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_unique_result_duplicate(self):
    self.assertRaises(exceptions.OSWinException, self.utils._unique_result, [mock.ANY, mock.ANY], mock.sentinel.resource_name)