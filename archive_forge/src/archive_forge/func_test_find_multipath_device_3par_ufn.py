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
def test_find_multipath_device_3par_ufn(self):

    def fake_execute(*cmd, **kwargs):
        out = "mpath6 (350002ac20398383d) dm-3 3PARdata,VV\nsize=2.0G features='0' hwhandler='0' wp=rw\n`-+- policy='round-robin 0' prio=-1 status=active\n  |- 0:0:0:1 sde 8:64 active undef running\n  `- 2:0:0:1 sdf 8:80 active undef running\n"
        return (out, None)
    self.linuxscsi._execute = fake_execute
    info = self.linuxscsi.find_multipath_device('/dev/sde')
    self.assertEqual('350002ac20398383d', info['id'])
    self.assertEqual('mpath6', info['name'])
    self.assertEqual('/dev/mapper/mpath6', info['device'])
    self.assertEqual('/dev/sde', info['devices'][0]['device'])
    self.assertEqual('0', info['devices'][0]['host'])
    self.assertEqual('0', info['devices'][0]['id'])
    self.assertEqual('0', info['devices'][0]['channel'])
    self.assertEqual('1', info['devices'][0]['lun'])
    self.assertEqual('/dev/sdf', info['devices'][1]['device'])
    self.assertEqual('2', info['devices'][1]['host'])
    self.assertEqual('0', info['devices'][1]['id'])
    self.assertEqual('0', info['devices'][1]['channel'])
    self.assertEqual('1', info['devices'][1]['lun'])