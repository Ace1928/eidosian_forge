import re
from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils10
def test_get_pci_device_address_error(self):
    self._check_get_pci_device_address_None(return_code=1)