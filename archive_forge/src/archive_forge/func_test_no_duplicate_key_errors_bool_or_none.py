from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_errors_bool_or_none(self):
    self.flakes('{True: 1, None: 2, False: 1}')