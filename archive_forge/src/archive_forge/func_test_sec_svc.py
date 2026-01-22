from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
def test_sec_svc(self):
    self._vmutils._sec_svc_attr = None
    self.assertEqual(self._vmutils._conn.Msvm_SecurityService.return_value[0], self._vmutils._sec_svc)
    self._vmutils._conn.Msvm_SecurityService.assert_called_with()