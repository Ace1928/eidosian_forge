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
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_sysfs_multipath_dm', side_effect=[None, 'dm-0'])
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwn', return_value='')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_add_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_add_wwid')
@mock.patch.object(iscsi.ISCSIConnector, '_connect_vol')
@mock.patch('os_brick.utils._time_sleep')
def test_connect_multipath_volume_no_wwid(self, sleep_mock, connect_mock, add_wwid_mock, add_path_mock, get_wwn_mock, find_dm_mock):

    def my_connect(rescans, props, data):
        devs = {'tgt1': 'sda', 'tgt2': 'sdb', 'tgt3': 'sdc', 'tgt4': 'sdd'}
        data['stopped_threads'] += 1
        data['num_logins'] += 1
        dev = devs[props['target_iqn']]
        data['found_devices'].append(dev)
        data['just_added_devices'].append(dev)
    connect_mock.side_effect = my_connect
    with mock.patch.object(self.connector, 'use_multipath'):
        res = self.connector._connect_multipath_volume(self.CON_PROPS)
    expected = {'type': 'block', 'scsi_wwn': '', 'multipath_id': '', 'path': '/dev/dm-0'}
    self.assertEqual(expected, res)
    self.assertEqual(3, get_wwn_mock.call_count)
    result = list(get_wwn_mock.call_args[0][0])
    result.sort()
    self.assertEqual(['sda', 'sdb', 'sdc', 'sdd'], result)
    mpath_values = [c[1][1] for c in get_wwn_mock._mock_mock_calls]
    self.assertEqual([None, None, 'dm-0'], mpath_values)
    add_wwid_mock.assert_not_called()
    add_path_mock.assert_not_called()
    self.assertGreaterEqual(find_dm_mock.call_count, 2)
    self.assertEqual(4, connect_mock.call_count)