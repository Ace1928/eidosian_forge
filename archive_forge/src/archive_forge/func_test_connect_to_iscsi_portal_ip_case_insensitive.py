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
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_sessions_full')
def test_connect_to_iscsi_portal_ip_case_insensitive(self, get_sessions_mock):
    """Connect creating node and session."""
    session = 'session2'
    get_sessions_mock.side_effect = [[('tcp:', 'session1', 'iP1:port1', '1', 'tgt')], [('tcp:', 'session1', 'Ip1:port1', '1', 'tgt'), ('tcp:', session, 'IP1:port1', '-1', 'tgt1')]]
    utils.ISCSI_SUPPORTS_MANUAL_SCAN = None
    with mock.patch.object(self.connector, '_execute') as exec_mock:
        exec_mock.side_effect = [('', 'error'), ('', None), ('', None), ('', None), ('', None)]
        res = self.connector._connect_to_iscsi_portal(self.CON_PROPS)
    self.assertEqual((session, True), res)
    self.assertTrue(utils.ISCSI_SUPPORTS_MANUAL_SCAN)
    prefix = 'iscsiadm -m node -T tgt1 -p ip1:port1'
    expected_cmds = [prefix, prefix + ' --interface default --op new', prefix + ' --op update -n node.session.scan -v manual', prefix + ' --login', prefix + ' --op update -n node.startup -v automatic']
    actual_cmds = [' '.join(args[0]) for args in exec_mock.call_args_list]
    self.assertListEqual(expected_cmds, actual_cmds)
    self.assertEqual(2, get_sessions_mock.call_count)