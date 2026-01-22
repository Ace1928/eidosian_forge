from unittest import mock
from os_brick.initiator import utils
from os_brick.tests import base
@mock.patch('os.name', 'nt')
def test_check_manual_scan_windows(self):
    self.assertFalse(utils.check_manual_scan())