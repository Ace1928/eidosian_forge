from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_hex_str_to_byte_array(self):
    fake_hex_str = '0x0010A'
    resulted_array = _utils.hex_str_to_byte_array(fake_hex_str)
    expected_array = bytearray([0, 1, 10])
    self.assertEqual(expected_array, resulted_array)