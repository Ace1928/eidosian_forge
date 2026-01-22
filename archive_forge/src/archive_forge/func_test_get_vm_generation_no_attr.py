from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_vm_generation_no_attr(self):
    mock_settings = self._lookup_vm()
    mock_settings.VirtualSystemSubType.side_effect = AttributeError
    ret = self._vmutils.get_vm_generation(mock.sentinel.FAKE_VM_NAME)
    self.assertEqual(constants.VM_GEN_1, ret)