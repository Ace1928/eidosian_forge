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
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwn', side_effect=(None, 'tgt2'))
@mock.patch.object(iscsi.ISCSIConnector, '_connect_vol')
@mock.patch.object(iscsi.ISCSIConnector, '_cleanup_connection')
@mock.patch('os_brick.utils._time_sleep')
def test_connect_single_volume(self, sleep_mock, cleanup_mock, connect_mock, get_wwn_mock):

    def my_connect(rescans, props, data):
        if props['target_iqn'] == 'tgt2':
            data['found_devices'].append('sdz')
    connect_mock.side_effect = my_connect
    res = self.connector._connect_single_volume(self.CON_PROPS)
    expected = {'type': 'block', 'scsi_wwn': 'tgt2', 'path': '/dev/sdz'}
    self.assertEqual(expected, res)
    get_wwn_mock.assert_has_calls([mock.call(['sdz']), mock.call(['sdz'])])
    sleep_mock.assert_called_once_with(1)
    cleanup_mock.assert_called_once_with({'target_lun': 4, 'volume_id': 'vol_id', 'target_portal': 'ip1:port1', 'target_iqn': 'tgt1'}, [('ip1:port1', 'tgt1', 4)], force=True, ignore_errors=True)