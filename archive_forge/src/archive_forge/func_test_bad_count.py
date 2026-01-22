import string
from taskflow import test
from taskflow.utils import iter_utils
def test_bad_count(self):
    self.assertRaises(ValueError, iter_utils.count, 2)