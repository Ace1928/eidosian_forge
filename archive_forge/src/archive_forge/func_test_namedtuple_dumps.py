from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
def test_namedtuple_dumps(self):
    for v in [Value(1), Point(1, 2), DuckValue(1), DuckPoint(1, 2)]:
        d = v._asdict()
        self.assertEqual(d, json.loads(json.dumps(v)))
        self.assertEqual(d, json.loads(json.dumps(v, namedtuple_as_object=True)))
        self.assertEqual(d, json.loads(json.dumps(v, tuple_as_array=False)))
        self.assertEqual(d, json.loads(json.dumps(v, namedtuple_as_object=True, tuple_as_array=False)))