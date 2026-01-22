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
def test_multipath_add_wwid(self):
    self.linuxscsi.multipath_add_wwid('wwid1')
    self.assertEqual(['multipath -a wwid1'], self.cmds)