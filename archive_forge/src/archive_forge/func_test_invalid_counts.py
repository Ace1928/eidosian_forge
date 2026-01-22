from unittest import TestCase
import simplejson as json
def test_invalid_counts(self):
    for n in ['foo', -1, 0, 1.0]:
        self.assertRaises(TypeError, json.dumps, 0, int_as_string_bitcount=n)