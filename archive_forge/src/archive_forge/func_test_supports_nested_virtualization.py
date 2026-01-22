from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_supports_nested_virtualization(self):
    self.assertFalse(self._hostutils.supports_nested_virtualization())