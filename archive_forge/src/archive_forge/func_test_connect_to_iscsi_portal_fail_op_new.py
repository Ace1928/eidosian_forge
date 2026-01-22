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
@mock.patch.object(iscsi.ISCSIConnector, '_iscsiadm_update')
@mock.patch.object(iscsi.ISCSIConnector, '_get_transport', return_value='default')
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_sessions_full')
@mock.patch('os_brick.utils._time_sleep')
def test_connect_to_iscsi_portal_fail_op_new(self, sleep_mock, get_sessions_mock, get_transport_mock, iscsiadm_update_mock):
    get_sessions_mock.return_value = []
    with mock.patch.object(self.connector, '_execute') as exec_mock:
        exec_mock.side_effect = [('', 21), ('', 6), ('', 21), ('', 6), ('', 21), ('', 6)]
        self.assertRaises(exception.BrickException, self.connector._connect_to_iscsi_portal, self.CON_PROPS)
    expected_cmds = ['iscsiadm -m node -T tgt1 -p ip1:port1', 'iscsiadm -m node -T tgt1 -p ip1:port1 --interface default --op new', 'iscsiadm -m node -T tgt1 -p ip1:port1', 'iscsiadm -m node -T tgt1 -p ip1:port1 --interface default --op new', 'iscsiadm -m node -T tgt1 -p ip1:port1', 'iscsiadm -m node -T tgt1 -p ip1:port1 --interface default --op new']
    actual_cmds = [' '.join(args[0]) for args in exec_mock.call_args_list]
    self.assertListEqual(expected_cmds, actual_cmds)
    iscsiadm_update_mock.assert_not_called()
    self.assertEqual(2, sleep_mock.call_count)