from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_required_vm_version(self):

    @_utils.required_vm_version()
    def foo(bar, vmsettings):
        pass
    mock_vmsettings = mock.Mock()
    for good_version in [constants.VM_VERSION_5_0, constants.VM_VERSION_254_0]:
        mock_vmsettings.Version = good_version
        foo(mock.sentinel.bar, mock_vmsettings)
    for bad_version in ['4.99', '254.1']:
        mock_vmsettings.Version = bad_version
        self.assertRaises(exceptions.InvalidVMVersion, foo, mock.sentinel.bar, mock_vmsettings)