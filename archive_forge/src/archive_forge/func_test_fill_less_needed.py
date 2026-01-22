import string
from taskflow import test
from taskflow.utils import iter_utils
def test_fill_less_needed(self):
    self.assertEqual('ab', ''.join(iter_utils.fill('abc', 2)))