from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def test_create_lookup_record_l2_only(self):
    self._check_create_lookup_record(constants.IPV4_DEFAULT, self.utils._LOOKUP_RECORD_TYPE_L2_ONLY)