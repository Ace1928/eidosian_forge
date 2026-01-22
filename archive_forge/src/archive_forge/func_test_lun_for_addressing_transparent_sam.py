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
@ddt.data(None, 'SAM', 'transparent')
def test_lun_for_addressing_transparent_sam(self, mode):
    lun = self.linuxscsi.lun_for_addressing(1, mode)
    self.assertEqual(1, lun)
    lun = self.linuxscsi.lun_for_addressing(256, mode)
    self.assertEqual(256, lun)