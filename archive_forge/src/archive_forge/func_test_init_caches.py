from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_init_caches(self):
    self.netutils._enable_cache = True
    self.netutils._switches = {}
    self.netutils._switch_ports = {}
    self.netutils._vlan_sds = {}
    self.netutils._profile_sds = {}
    self.netutils._hw_offload_sds = {}
    self.netutils._vsid_sds = {}
    self.netutils._bandwidth_sds = {}
    conn = self.netutils._conn
    mock_vswitch = mock.MagicMock(ElementName=mock.sentinel.vswitch_name)
    conn.Msvm_VirtualEthernetSwitch.return_value = [mock_vswitch]
    mock_port = mock.MagicMock(ElementName=mock.sentinel.port_name)
    conn.Msvm_EthernetPortAllocationSettingData.return_value = [mock_port]
    mock_sd = mock.MagicMock(InstanceID=self._FAKE_INSTANCE_ID)
    mock_bad_sd = mock.MagicMock(InstanceID=self._FAKE_BAD_INSTANCE_ID)
    conn.Msvm_EthernetSwitchPortProfileSettingData.return_value = [mock_bad_sd, mock_sd]
    conn.Msvm_EthernetSwitchPortVlanSettingData.return_value = [mock_bad_sd, mock_sd]
    conn.Msvm_EthernetSwitchPortSecuritySettingData.return_value = [mock_bad_sd, mock_sd]
    conn.Msvm_EthernetSwitchPortBandwidthSettingData.return_value = [mock_bad_sd, mock_sd]
    conn.Msvm_EthernetSwitchPortOffloadSettingData.return_value = [mock_bad_sd, mock_sd]
    self.netutils.init_caches()
    self.assertEqual({mock.sentinel.vswitch_name: mock_vswitch}, self.netutils._switches)
    self.assertEqual({mock.sentinel.port_name: mock_port}, self.netutils._switch_ports)
    self.assertEqual([mock_sd], list(self.netutils._profile_sds.values()))
    self.assertEqual([mock_sd], list(self.netutils._vlan_sds.values()))
    self.assertEqual([mock_sd], list(self.netutils._vsid_sds.values()))
    self.assertEqual([mock_sd], list(self.netutils._bandwidth_sds.values()))
    self.assertEqual([mock_sd], list(self.netutils._hw_offload_sds.values()))