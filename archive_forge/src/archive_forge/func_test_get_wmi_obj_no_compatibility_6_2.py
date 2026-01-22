from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils.BaseUtilsVirt, '_get_wmi_compat_conn')
def test_get_wmi_obj_no_compatibility_6_2(self, mock_get_wmi_compat):
    baseutils.BaseUtilsVirt._os_version = [6, 2]
    result = self.utils._get_wmi_obj(mock.sentinel.moniker, False)
    self.assertEqual(self._mock_wmi.WMI.return_value, result)