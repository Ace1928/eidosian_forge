from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_remove_virt_feature(self):
    self._test_virt_method('RemoveFeatureSettings', 2, 'remove_virt_feature', False, FeatureSettings=[mock.sentinel.res_path])