import errno
from unittest import mock
from os_brick.initiator import host_driver
from os_brick.tests import base
def test_get_all_block_devices(self):
    fake_dev = ['device1', 'device2']
    expected = ['/dev/disk/by-path/' + dev for dev in fake_dev]
    driver = host_driver.HostDriver()
    with mock.patch('os.listdir', return_value=fake_dev):
        actual = driver.get_all_block_devices()
    self.assertEqual(expected, actual)