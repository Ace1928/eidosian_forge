from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_keys_ints(self):
    self.flakes('{1: 1, 1: 2}', m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)