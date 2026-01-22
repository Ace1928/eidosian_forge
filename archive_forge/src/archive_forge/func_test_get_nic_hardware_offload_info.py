from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
@mock.patch.object(hostutils.HostUtils, '_get_nic_hw_offload_info')
def test_get_nic_hardware_offload_info(self, mock_get_nic_offload):
    mock_vswitch_sd = mock.Mock(VirtualSystemIdentifier=mock.sentinel.vsid)
    mock_hw_offload_sd = mock.Mock(SystemName=mock.sentinel.vsid)
    vswitch_sds_class = self._conn.Msvm_VirtualEthernetSwitchSettingData
    vswitch_sds_class.return_value = [mock_vswitch_sd]
    hw_offload_class = self._conn.Msvm_EthernetSwitchHardwareOffloadData
    hw_offload_class.return_value = [mock_hw_offload_sd]
    hw_offload_info = self._hostutils.get_nic_hardware_offload_info()
    self.assertEqual([mock_get_nic_offload.return_value], hw_offload_info)
    vswitch_sds_class.assert_called_once_with()
    hw_offload_class.assert_called_once_with()
    mock_get_nic_offload.assert_called_once_with(mock_vswitch_sd, mock_hw_offload_sd)