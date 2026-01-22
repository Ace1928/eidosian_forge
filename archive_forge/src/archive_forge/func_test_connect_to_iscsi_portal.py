from unittest import mock
from os_brick.initiator.connectors import iscsi
from os_brick.tests.initiator import test_connector
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_sessions_full')
@mock.patch.object(iscsi.ISCSIConnector, '_execute')
def test_connect_to_iscsi_portal(self, exec_mock, sessions_mock):
    """Connect to portal while session already established"""
    sessions_mock.side_effect = [[('iser:', 'session_iser', 'ip:port', '1', 'target_1')]]
    exec_mock.side_effect = [('', None), ('', None), ('', None)]
    res = self.connector._connect_to_iscsi_portal(self.connection_data)
    self.assertEqual(('session_iser', True), res)
    prefix = 'iscsiadm -m node -T target_1 -p ip:port'
    expected_cmds = [prefix, prefix + ' --op update -n node.session.scan -v manual']
    actual_cmds = [' '.join(args[0]) for args in exec_mock.call_args_list]
    self.assertListEqual(expected_cmds, actual_cmds)