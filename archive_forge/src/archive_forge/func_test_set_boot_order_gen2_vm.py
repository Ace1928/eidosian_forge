from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_set_boot_order_gen2_vm(self):
    self._test_set_boot_order(vm_gen=constants.VM_GEN_2)