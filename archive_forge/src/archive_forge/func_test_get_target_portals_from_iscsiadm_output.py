import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
def test_get_target_portals_from_iscsiadm_output(self):
    connector = self.connector
    test_output = '10.15.84.19:3260,1 iqn.1992-08.com.netapp:sn.33615311\n                         10.15.85.19:3260,2 iqn.1992-08.com.netapp:sn.33615311\n                         '
    res = connector._get_target_portals_from_iscsiadm_output(test_output)
    ips = ['10.15.84.19:3260', '10.15.85.19:3260']
    iqns = ['iqn.1992-08.com.netapp:sn.33615311', 'iqn.1992-08.com.netapp:sn.33615311']
    expected = (ips, iqns)
    self.assertEqual(expected, res)