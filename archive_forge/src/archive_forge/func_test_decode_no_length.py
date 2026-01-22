import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_decode_no_length(self):
    self.assertDecode(0, 65536, 1, b'\x80', 0)
    self.assertDecode(1, 65536, 2, b'\x81\x01', 0)
    self.assertDecode(10, 65536, 2, b'\x81\n', 0)
    self.assertDecode(255, 65536, 2, b'\x81\xff', 0)
    self.assertDecode(256, 65536, 2, b'\x82\x01', 0)
    self.assertDecode(257, 65536, 3, b'\x83\x01\x01', 0)
    self.assertDecode(4294967295, 65536, 5, b'\x8f\xff\xff\xff\xff', 0)
    self.assertDecode(4294967040, 65536, 4, b'\x8e\xff\xff\xff', 0)
    self.assertDecode(4294902015, 65536, 4, b'\x8d\xff\xff\xff', 0)
    self.assertDecode(4278255615, 65536, 4, b'\x8b\xff\xff\xff', 0)
    self.assertDecode(16777215, 65536, 4, b'\x87\xff\xff\xff', 0)
    self.assertDecode(16909060, 65536, 5, b'\x8f\x04\x03\x02\x01', 0)