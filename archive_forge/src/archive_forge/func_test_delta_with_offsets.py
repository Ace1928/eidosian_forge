import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_delta_with_offsets(self):
    di = self._gc_module.DeltaIndex()
    di.add_source(_first_text, 5)
    self.assertEqual(len(_first_text) + 5, di._source_offset)
    di.add_source(_second_text, 10)
    self.assertEqual(len(_first_text) + len(_second_text) + 15, di._source_offset)
    delta = di.make_delta(_third_text)
    self.assertIsNot(None, delta)
    result = self._gc_module.apply_delta(b'12345' + _first_text + b'1234567890' + _second_text, delta)
    self.assertIsNot(None, result)
    self.assertEqualDiff(_third_text, result)
    self.assertEqual(b'\x85\x01\x91\x05\x14\x0chas some in \x91\x856\x03and\x91s"\x91?\n', delta)