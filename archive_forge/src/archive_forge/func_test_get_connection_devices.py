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
@mock.patch.object(iscsi.ISCSIConnector, '_get_ips_iqns_luns')
@mock.patch('glob.glob')
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_sessions_full')
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_nodes')
def test_get_connection_devices(self, nodes_mock, sessions_mock, glob_mock, iql_mock):
    iql_mock.return_value = self.connector._get_all_targets(self.CON_PROPS)
    sessions_mock.return_value = [('non-tcp:', '0', 'ip1:port1', '1', 'tgt1'), ('tcp:', '1', 'ip1:port1', '1', 'tgt1'), ('tcp:', '2', 'ip2:port2', '-1', 'tgt2'), ('tcp:', '3', 'ip1:port1', '1', 'tgt4'), ('tcp:', '4', 'ip2:port2', '-1', 'tgt5')]
    nodes_mock.return_value = [('ip1:port1', 'tgt1'), ('ip2:port2', 'tgt2'), ('ip3:port3', 'tgt3')]
    sys_cls = '/sys/class/scsi_host/host'
    glob_mock.side_effect = [[sys_cls + '1/device/session1/target6/1:2:6:4/block/sda', sys_cls + '1/device/session1/target6/1:2:6:4/block/sda1'], [sys_cls + '2/device/session2/target7/2:2:7:5/block/sdb', sys_cls + '2/device/session2/target7/2:2:7:4/block/sdc']]
    res = self.connector._get_connection_devices(self.CON_PROPS)
    expected = {('ip1:port1', 'tgt1'): ({'sda'}, set()), ('ip2:port2', 'tgt2'): ({'sdb'}, {'sdc'}), ('ip3:port3', 'tgt3'): (set(), set())}
    self.assertDictEqual(expected, res)
    iql_mock.assert_called_once_with(self.CON_PROPS, discover=False, is_disconnect_call=False)