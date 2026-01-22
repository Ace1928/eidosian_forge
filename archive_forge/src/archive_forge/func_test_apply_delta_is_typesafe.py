import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_apply_delta_is_typesafe(self):
    self.apply_delta(_text1, b'M\x90M')
    self.assertRaises(TypeError, self.apply_delta, object(), b'M\x90M')
    self.assertRaises(TypeError, self.apply_delta, _text1.decode('latin1'), b'M\x90M')
    self.assertRaises(TypeError, self.apply_delta, _text1, 'M\x90M')
    self.assertRaises(TypeError, self.apply_delta, _text1, object())