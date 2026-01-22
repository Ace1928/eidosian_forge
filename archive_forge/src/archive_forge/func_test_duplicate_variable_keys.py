from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_variable_keys(self):
    self.flakes('\n            a = 1\n            {a: 1, a: 2}\n            ', m.MultiValueRepeatedKeyVariable, m.MultiValueRepeatedKeyVariable)