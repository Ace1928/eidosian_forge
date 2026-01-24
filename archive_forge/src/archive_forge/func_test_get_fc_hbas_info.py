import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_hbas')
def test_get_fc_hbas_info(self, mock_hbas):
    host_pci = '/sys/devices/css0/0.0.02ea/0.0.3080/host0/fc_host/host0'
    mock_hbas.return_value = [{'ClassDevice': 'host0', 'ClassDevicepath': host_pci, 'port_name': '0xc05076ffe680a960', 'node_name': '0x1234567898928432', 'port_state': 'Online'}]
    hbas_info = self.lfc.get_fc_hbas_info()
    expected = [{'device_path': '/sys/devices/css0/0.0.02ea/0.0.3080/host0/fc_host/host0', 'host_device': 'host0', 'node_name': '1234567898928432', 'port_name': 'c05076ffe680a960'}]
    self.assertEqual(expected, hbas_info)