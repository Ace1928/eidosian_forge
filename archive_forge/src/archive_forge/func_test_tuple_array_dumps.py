import unittest
from simplejson.compat import StringIO
import simplejson as json
def test_tuple_array_dumps(self):
    t = (1, 2, 3)
    expect = json.dumps(list(t))
    self.assertEqual(expect, json.dumps(t))
    self.assertEqual(expect, json.dumps(t, tuple_as_array=True))
    self.assertRaises(TypeError, json.dumps, t, tuple_as_array=False)
    self.assertEqual(expect, json.dumps(t, default=repr))
    self.assertEqual(expect, json.dumps(t, tuple_as_array=True, default=repr))
    self.assertEqual(json.dumps(repr(t)), json.dumps(t, tuple_as_array=False, default=repr))