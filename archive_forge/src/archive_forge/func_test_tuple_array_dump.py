import unittest
from simplejson.compat import StringIO
import simplejson as json
def test_tuple_array_dump(self):
    t = (1, 2, 3)
    expect = json.dumps(list(t))
    sio = StringIO()
    json.dump(t, sio)
    self.assertEqual(expect, sio.getvalue())
    sio = StringIO()
    json.dump(t, sio, tuple_as_array=True)
    self.assertEqual(expect, sio.getvalue())
    self.assertRaises(TypeError, json.dump, t, StringIO(), tuple_as_array=False)
    sio = StringIO()
    json.dump(t, sio, default=repr)
    self.assertEqual(expect, sio.getvalue())
    sio = StringIO()
    json.dump(t, sio, tuple_as_array=True, default=repr)
    self.assertEqual(expect, sio.getvalue())
    sio = StringIO()
    json.dump(t, sio, tuple_as_array=False, default=repr)
    self.assertEqual(json.dumps(repr(t)), sio.getvalue())