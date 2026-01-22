import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_encode_no_offset(self):
    self.assertEncode(b'\x90\x01', 0, 1)
    self.assertEncode(b'\x90\n', 0, 10)
    self.assertEncode(b'\x90\xff', 0, 255)
    self.assertEncode(b'\xa0\x01', 0, 256)
    self.assertEncode(b'\xb0\x01\x01', 0, 257)
    self.assertEncode(b'\xb0\xff\xff', 0, 65535)
    self.assertEncode(b'\x80', 0, 64 * 1024)