import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch('builtins.open')
@ddt.data(True, False)
def test__get_hba_channel_scsi_target_lun_with_initiator_target_map(self, remote_scan, mock_open):
    execute_results, expected_cmds = self._get_expected_info(wwpns=['514f0c50023f6c01'])
    if remote_scan:
        mock_open = mock_open.return_value.__enter__.return_value
        mock_open.read.return_value = '1\n'
    hbas, con_props = self.__get_rescan_info(zone_manager=True)
    con_props['target_wwn'] = con_props['target_wwn'][0]
    con_props['targets'] = con_props['targets'][0:1]
    hbas[0]['port_name'] = '50014380186af83e'
    with mock.patch.object(self.lfc, '_execute', side_effect=execute_results) as execute_mock:
        res = self.lfc._get_hba_channel_scsi_target_lun(hbas[0], con_props)
        execute_mock.assert_has_calls(expected_cmds)
    expected = ([['0', '1', 1]], set())
    self.assertEqual(expected, res)