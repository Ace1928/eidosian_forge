from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_set_vm_vcpus_per_vnuma_node(self):
    self._check_set_vm_vcpus(vcpus_per_numa_node=1)