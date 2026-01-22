from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, 'get_vm_generation')
def test_vm_has_s3_controller(self, mock_get_vm_generation):
    self.assertTrue(self._vmutils._vm_has_s3_controller(mock.sentinel.fake_vm_name))