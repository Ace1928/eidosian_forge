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
def test_process_lun_id_single_val_make_hex(self):
    lun_id = 499
    result = self.linuxscsi.process_lun_id(lun_id)
    expected = '0x01f3000000000000'
    self.assertEqual(expected, result)