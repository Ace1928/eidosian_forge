import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
def test_finditer(self):
    pattern = lazy_regex.lazy_compile('fo*')
    matches = [(m.start(), m.end(), m.group()) for m in pattern.finditer('foo bar fop')]
    self.assertEqual([(0, 3, 'foo'), (8, 10, 'fo')], matches)