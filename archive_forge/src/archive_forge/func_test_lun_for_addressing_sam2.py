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
@ddt.data((1, 1), (100, 100), (256, 16640), (1010, 17394))
@ddt.unpack
def test_lun_for_addressing_sam2(self, original_lun, expected_lun):
    lun = self.linuxscsi.lun_for_addressing(original_lun, 'SAM2')
    self.assertEqual(expected_lun, lun)