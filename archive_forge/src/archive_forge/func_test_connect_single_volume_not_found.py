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
def test_connect_single_volume_not_found(self, sleep_mock, cleanup_mock, connect_mock, get_wwn_mock):
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector._connect_single_volume, self.CON_PROPS)
    get_wwn_mock.assert_not_called()
    self.assertEqual(2, sleep_mock.call_count)
    props = list(self.connector._get_all_targets(self.CON_PROPS))
    calls_per_try = [mock.call({'target_portal': prop[0], 'target_iqn': prop[1], 'target_lun': prop[2], 'volume_id': 'vol_id'}, [prop], force=True, ignore_errors=True) for prop in props]
    cleanup_mock.assert_has_calls(calls_per_try * 3)
    data = self._get_connect_vol_data()
    calls_per_try = [mock.call(self.connector.device_scan_attempts, {'target_portal': prop[0], 'target_iqn': prop[1], 'target_lun': prop[2], 'volume_id': 'vol_id'}, data) for prop in props]
    connect_mock.assert_has_calls(calls_per_try * 3)