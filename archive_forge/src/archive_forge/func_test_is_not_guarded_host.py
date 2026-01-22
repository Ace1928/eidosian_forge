import re
from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils10
def test_is_not_guarded_host(self):
    self._test_is_host_guarded(is_host_guarded=False)