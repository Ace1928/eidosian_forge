from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
def test_asdict_not_callable_dumps(self):
    for f in CONSTRUCTORS:
        self.assertRaises(TypeError, json.dumps, f(DeadDuck()), namedtuple_as_object=True)
        self.assertRaises(TypeError, json.dumps, f(Value), namedtuple_as_object=True)
        self.assertEqual(json.dumps(f({})), json.dumps(f(DeadDict()), namedtuple_as_object=True))