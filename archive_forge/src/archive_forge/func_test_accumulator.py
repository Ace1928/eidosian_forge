from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_accumulator(self):
    lst = [0] * 100000
    self.assertEqual(json.loads(json.dumps(lst)), lst)