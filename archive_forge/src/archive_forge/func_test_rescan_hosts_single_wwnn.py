import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch.object(linuxfc.LinuxFibreChannel, 'lun_for_addressing')
def test_rescan_hosts_single_wwnn(self, lun_addr_mock):
    """Test FC rescan with no initiator map and single WWNN for ports."""
    lun_addr_mock.return_value = 16640
    get_chan_results = [[[['2', '3', 256], ['4', '5', 256]], set()], [[['6', '7', 256]], set()], [[], {1}]]
    hbas, con_props = self.__get_rescan_info(zone_manager=False)
    con_props['addressing_mode'] = 'SAM2'
    hbas.append({'device_path': '/sys/devices/pci0000:00/0000:00:02.0/0000:04:00.2/host8/fc_host/host8', 'host_device': 'host8', 'node_name': '50014380186af83g', 'port_name': '50014380186af83h'})
    with mock.patch.object(self.lfc, '_execute', return_value=None) as execute_mock, mock.patch.object(self.lfc, '_get_hba_channel_scsi_target_lun', side_effect=get_chan_results) as mock_get_chan:
        self.lfc.rescan_hosts(hbas, con_props)
        expected_commands = [mock.call('tee', '-a', '/sys/class/scsi_host/host6/scan', process_input='2 3 16640', root_helper=None, run_as_root=True), mock.call('tee', '-a', '/sys/class/scsi_host/host6/scan', process_input='4 5 16640', root_helper=None, run_as_root=True), mock.call('tee', '-a', '/sys/class/scsi_host/host7/scan', process_input='6 7 16640', root_helper=None, run_as_root=True)]
        execute_mock.assert_has_calls(expected_commands)
        self.assertEqual(len(expected_commands), execute_mock.call_count)
        expected_calls = [mock.call(hbas[0], con_props), mock.call(hbas[1], con_props)]
        mock_get_chan.assert_has_calls(expected_calls)
    lun_addr_mock.assert_has_calls([mock.call(256, 'SAM2')] * 3)