import re
from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils10
def test_get_pci_device_address(self):
    pnp_device = mock.MagicMock()
    pnp_device_properties = [mock.MagicMock(KeyName='DEVPKEY_Device_LocationInfo', Data='bus 2, domain 4, function 0'), mock.MagicMock(KeyName='DEVPKEY_Device_Address', Data=0)]
    pnp_device.GetDeviceProperties.return_value = (0, pnp_device_properties)
    self._hostutils._conn_cimv2.Win32_PnPEntity.return_value = [pnp_device]
    result = self._hostutils._get_pci_device_address(mock.sentinel.device_instance_path)
    pnp_props = {prop.KeyName: prop.Data for prop in pnp_device_properties}
    location_info = pnp_props['DEVPKEY_Device_LocationInfo']
    slot = pnp_props['DEVPKEY_Device_Address']
    [bus, domain, function] = re.findall('\\b\\d+\\b', location_info)
    expected_result = '%04x:%02x:%02x.%1x' % (int(domain), int(bus), int(slot), int(function))
    self.assertEqual(expected_result, result)
    self._hostutils._conn_cimv2.Win32_PnPEntity.assert_called_once_with(DeviceID=mock.sentinel.device_instance_path)