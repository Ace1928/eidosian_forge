import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
def test_simple_acts_like_regex(self):
    """Test that the returned object has basic regex like functionality"""
    pattern = lazy_regex.lazy_compile('foo')
    self.assertIsInstance(pattern, lazy_regex.LazyRegex)
    self.assertTrue(pattern.match('foo'))
    self.assertIs(None, pattern.match('bar'))