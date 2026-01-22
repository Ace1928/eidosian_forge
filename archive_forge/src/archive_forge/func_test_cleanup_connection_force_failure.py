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
@mock.patch('os_brick.exception.ExceptionChainer.__nonzero__', mock.Mock(return_value=True))
@mock.patch('os_brick.exception.ExceptionChainer.__bool__', mock.Mock(return_value=True))
@mock.patch.object(iscsi.ISCSIConnector, '_disconnect_connection')
@mock.patch.object(iscsi.ISCSIConnector, '_get_connection_devices')
@mock.patch.object(linuxscsi.LinuxSCSI, 'flush_multipath_device')
@mock.patch.object(linuxscsi.LinuxSCSI, 'remove_connection', return_value=mock.sentinel.mp_name)
def test_cleanup_connection_force_failure(self, remove_mock, flush_mock, con_devs_mock, discon_mock):
    con_devs_mock.return_value = collections.OrderedDict(((('ip1:port1', 'tgt1'), ({'sda'}, set())), (('ip2:port2', 'tgt2'), ({'sdb'}, {'sdc'})), (('ip3:port3', 'tgt3'), (set(), set()))))
    self.assertRaises(exception.ExceptionChainer, self.connector._cleanup_connection, self.CON_PROPS, ips_iqns_luns=mock.sentinel.ips_iqns_luns, force=mock.sentinel.force, ignore_errors=False)
    con_devs_mock.assert_called_once_with(self.CON_PROPS, mock.sentinel.ips_iqns_luns, False)
    remove_mock.assert_called_once_with({'sda', 'sdb'}, mock.sentinel.force, mock.ANY, '', False)
    discon_mock.assert_called_once_with(self.CON_PROPS, [('ip1:port1', 'tgt1'), ('ip3:port3', 'tgt3')], mock.sentinel.force, mock.ANY)
    flush_mock.assert_called_once_with(mock.sentinel.mp_name)