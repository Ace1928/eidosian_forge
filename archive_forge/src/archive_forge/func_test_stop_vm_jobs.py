from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_stop_vm_jobs(self):
    mock_vm = self._lookup_vm()
    self._vmutils.stop_vm_jobs(mock.sentinel.vm_name)
    self._vmutils._jobutils.stop_jobs.assert_called_once_with(mock_vm, None)