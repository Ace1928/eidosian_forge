from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_errors(self):
    self.flakes("\n        {'yes': 1, 'no': 2}\n        ")