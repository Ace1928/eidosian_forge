import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_delta_against_multiple_sources(self):
    di = self._gc_module.DeltaIndex()
    di.add_source(_first_text, 0)
    self.assertEqual(len(_first_text), di._source_offset)
    di.add_source(_second_text, 0)
    self.assertEqual(len(_first_text) + len(_second_text), di._source_offset)
    delta = di.make_delta(_third_text)
    result = self._gc_module.apply_delta(_first_text + _second_text, delta)
    self.assertEqualDiff(_third_text, result)
    self.assertEqual(b'\x85\x01\x90\x14\x0chas some in \x91v6\x03and\x91d"\x91:\n', delta)