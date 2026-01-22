from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def test_get_portal_locations_failing_if_none(self):
    self._test_get_portal_locations(available_only=True, fail_if_none_found=True)