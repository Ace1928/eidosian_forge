from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_keys_bools_false(self):
    self.flakes('{False: 1, False: 2}', m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)