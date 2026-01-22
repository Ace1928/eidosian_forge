from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_vm_summary_info(self):
    self._lookup_vm()
    mock_summary = mock.MagicMock()
    mock_svc = self._vmutils._vs_man_svc
    mock_svc.GetSummaryInformation.return_value = (self._FAKE_RET_VAL, [mock_summary])
    for key, val in self._FAKE_SUMMARY_INFO.items():
        setattr(mock_summary, key, val)
    summary = self._vmutils.get_vm_summary_info(self._FAKE_VM_NAME)
    self.assertEqual(self._FAKE_SUMMARY_INFO, summary)