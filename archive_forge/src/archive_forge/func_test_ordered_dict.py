from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_ordered_dict(self):
    items = [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)]
    s = json.dumps(json.OrderedDict(items))
    self.assertEqual(s, '{"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}')