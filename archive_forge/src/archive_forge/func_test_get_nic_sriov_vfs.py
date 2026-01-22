from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_nic_sriov_vfs(self):
    mock_vswitch_sd = mock.Mock()
    mock_hw_offload_sd_bad = mock.Mock(IovVfCapacity=0)
    mock_hw_offload_sd_ok = mock.Mock()
    vswitch_sds_class = self._conn.Msvm_VirtualEthernetSwitchSettingData
    vswitch_sds_class.return_value = [mock_vswitch_sd] * 3
    self._conn.Msvm_EthernetSwitchHardwareOffloadData.side_effect = [[mock_hw_offload_sd_bad], [mock_hw_offload_sd_ok], [mock_hw_offload_sd_ok]]
    self._netutils.get_vswitch_external_network_name.side_effect = [None, mock.sentinel.nic_name]
    mock_nic = mock.Mock()
    self._conn_scimv2.MSFT_NetAdapter.return_value = [mock_nic]
    vfs = self._hostutils.get_nic_sriov_vfs()
    expected = {'vswitch_name': mock_vswitch_sd.ElementName, 'device_id': mock_nic.PnPDeviceID, 'total_vfs': mock_hw_offload_sd_ok.IovVfCapacity, 'used_vfs': mock_hw_offload_sd_ok.IovVfUsage}
    self.assertEqual([expected], vfs)
    vswitch_sds_class.assert_called_once_with(IOVPreferred=True)
    self._conn.Msvm_EthernetSwitchHardwareOffloadData.assert_has_calls([mock.call(SystemName=mock_vswitch_sd.VirtualSystemIdentifier)] * 3)
    self._netutils.get_vswitch_external_network_name.assert_has_calls([mock.call(mock_vswitch_sd.ElementName)] * 2)
    self._conn_scimv2.MSFT_NetAdapter.assert_called_once_with(InterfaceDescription=mock.sentinel.nic_name)