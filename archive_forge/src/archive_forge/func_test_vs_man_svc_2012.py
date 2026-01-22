from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils, 'wmi', create=True)
def test_vs_man_svc_2012(self, mock_wmi):
    baseutils.BaseUtilsVirt._old_wmi = None
    mock_os = mock.MagicMock(Version='6.2.0')
    mock_wmi.WMI.return_value.Win32_OperatingSystem.return_value = [mock_os]
    fake_module_path = '/fake/path/to/module'
    mock_wmi.__path__ = [fake_module_path]
    spec = importlib.util.spec_from_file_location.return_value
    module = importlib.util.module_from_spec.return_value
    old_conn = module.WMI.return_value
    expected = old_conn.Msvm_VirtualSystemManagementService()[0]
    self.assertEqual(expected, self.utils._vs_man_svc)
    self.assertIsNone(self.utils._vs_man_svc_attr)
    importlib.util.spec_from_file_location.assert_called_once_with('old_wmi', '%s.py' % fake_module_path)
    spec.loader.exec_module.assert_called_once_with(module)
    importlib.util.module_from_spec.assert_called_once_with(importlib.util.spec_from_file_location.return_value)