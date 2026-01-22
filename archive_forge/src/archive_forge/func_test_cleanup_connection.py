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
@ddt.data(('/dev/sda', False), ('/dev/disk/by-id/scsi-WWID', False), ('/dev/dm-11', True), ('/dev/disk/by-id/dm-uuid-mpath-MPATH', True))
@ddt.unpack
@mock.patch('os_brick.utils.get_dev_path')
@mock.patch.object(iscsi.ISCSIConnector, '_disconnect_connection')
@mock.patch.object(iscsi.ISCSIConnector, '_get_connection_devices')
@mock.patch.object(linuxscsi.LinuxSCSI, 'flush_multipath_device')
@mock.patch.object(linuxscsi.LinuxSCSI, 'remove_connection', return_value=None)
def test_cleanup_connection(self, path_used, was_multipath, remove_mock, flush_mock, con_devs_mock, discon_mock, get_dev_path_mock):
    get_dev_path_mock.return_value = path_used
    con_devs_mock.return_value = collections.OrderedDict(((('ip1:port1', 'tgt1'), ({'sda'}, set())), (('ip2:port2', 'tgt2'), ({'sdb'}, {'sdc'})), (('ip3:port3', 'tgt3'), (set(), set()))))
    self.connector._cleanup_connection(self.CON_PROPS, ips_iqns_luns=mock.sentinel.ips_iqns_luns, force=False, ignore_errors=False, device_info=mock.sentinel.device_info)
    get_dev_path_mock.assert_called_once_with(self.CON_PROPS, mock.sentinel.device_info)
    con_devs_mock.assert_called_once_with(self.CON_PROPS, mock.sentinel.ips_iqns_luns, False)
    remove_mock.assert_called_once_with({'sda', 'sdb'}, False, mock.ANY, path_used, was_multipath)
    discon_mock.assert_called_once_with(self.CON_PROPS, [('ip1:port1', 'tgt1'), ('ip3:port3', 'tgt3')], False, mock.ANY)
    flush_mock.assert_not_called()