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
@mock.patch('os_brick.utils._time_sleep')
@mock.patch.object(linuxscsi.LinuxSCSI, 'scan_iscsi')
@mock.patch.object(linuxscsi.LinuxSCSI, 'device_name_by_hctl', return_value='sda')
@mock.patch.object(iscsi.ISCSIConnector, '_connect_to_iscsi_portal')
def test_connect_vol(self, connect_mock, dev_name_mock, scan_mock, sleep_mock):
    lscsi = self.connector._linuxscsi
    data = self._get_connect_vol_data()
    hctl = [mock.sentinel.host, mock.sentinel.channel, mock.sentinel.target, mock.sentinel.lun]
    connect_mock.return_value = (mock.sentinel.session, False)
    with mock.patch.object(lscsi, 'get_hctl', side_effect=(None, hctl)) as hctl_mock:
        self.connector._connect_vol(3, self.CON_PROPS, data)
    expected = self._get_connect_vol_data()
    expected.update(num_logins=1, stopped_threads=1, found_devices=['sda'], just_added_devices=['sda'])
    self.assertDictEqual(expected, data)
    connect_mock.assert_called_once_with(self.CON_PROPS)
    hctl_mock.assert_has_calls([mock.call(mock.sentinel.session, self.CON_PROPS['target_lun']), mock.call(mock.sentinel.session, self.CON_PROPS['target_lun'])])
    scan_mock.assert_not_called()
    dev_name_mock.assert_called_once_with(mock.sentinel.session, hctl)
    sleep_mock.assert_called_once_with(1)