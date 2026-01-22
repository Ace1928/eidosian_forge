from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_values_bytes_vs_unicode_py3(self):
    self.flakes("{1: b'a', 1: u'a'}", m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)