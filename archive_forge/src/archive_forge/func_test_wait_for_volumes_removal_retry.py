import os
import os.path
import textwrap
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.initiator import linuxscsi
from os_brick.tests import base
@mock.patch('os_brick.utils._time_sleep')
@mock.patch('os.path.exists', side_effect=(True, True, False, False))
def test_wait_for_volumes_removal_retry(self, exists_mock, sleep_mock):
    names = ('sda', 'sdb')
    self.linuxscsi.wait_for_volumes_removal(names)
    exists_mock.assert_has_calls([mock.call('/dev/' + name) for name in names] * 2)
    self.assertEqual(1, sleep_mock.call_count)