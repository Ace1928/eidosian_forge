from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import fibre_channel_s390x
from os_brick.initiator import linuxfc
from os_brick.tests.initiator import test_connector
@mock.patch.object(linuxfc.LinuxFibreChannelS390X, 'configure_scsi_device')
def test_get_host_devices(self, mock_configure_scsi_device):
    possible_devs = [(3, 5, 2)]
    devices = self.connector._get_host_devices(possible_devs)
    mock_configure_scsi_device.assert_called_with(3, 5, '0x0002000000000000')
    self.assertEqual(3, len(devices))
    device_path = '/dev/disk/by-path/ccw-3-zfcp-5:0x0002000000000000'
    self.assertEqual(devices[0], device_path)
    device_path = '/dev/disk/by-path/ccw-3-fc-5-lun-2'
    self.assertEqual(devices[1], device_path)
    device_path = '/dev/disk/by-path/ccw-3-fc-5-lun-0x0002000000000000'
    self.assertEqual(devices[2], device_path)