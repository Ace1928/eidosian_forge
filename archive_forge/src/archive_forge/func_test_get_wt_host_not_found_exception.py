from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def test_get_wt_host_not_found_exception(self):
    self._test_get_wt_host(host_found=False, fail_if_not_found=True)