from unittest import TestCase
import simplejson.encoder
from simplejson.compat import b
def test_sorted_dict(self):
    items = [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)]
    s = simplejson.dumps(dict(items), sort_keys=True)
    self.assertEqual(s, '{"five": 5, "four": 4, "one": 1, "three": 3, "two": 2}')