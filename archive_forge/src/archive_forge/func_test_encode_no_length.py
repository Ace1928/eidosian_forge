import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_encode_no_length(self):
    self.assertEncode(b'\x80', 0, 64 * 1024)
    self.assertEncode(b'\x81\x01', 1, 64 * 1024)
    self.assertEncode(b'\x81\n', 10, 64 * 1024)
    self.assertEncode(b'\x81\xff', 255, 64 * 1024)
    self.assertEncode(b'\x82\x01', 256, 64 * 1024)
    self.assertEncode(b'\x83\x01\x01', 257, 64 * 1024)
    self.assertEncode(b'\x8f\xff\xff\xff\xff', 4294967295, 64 * 1024)
    self.assertEncode(b'\x8e\xff\xff\xff', 4294967040, 64 * 1024)
    self.assertEncode(b'\x8d\xff\xff\xff', 4294902015, 64 * 1024)
    self.assertEncode(b'\x8b\xff\xff\xff', 4278255615, 64 * 1024)
    self.assertEncode(b'\x87\xff\xff\xff', 16777215, 64 * 1024)
    self.assertEncode(b'\x8f\x04\x03\x02\x01', 16909060, 64 * 1024)