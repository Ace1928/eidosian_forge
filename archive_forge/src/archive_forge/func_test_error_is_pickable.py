import sys, pickle
from unittest import TestCase
import simplejson as json
from simplejson.compat import text_type, b
def test_error_is_pickable(self):
    err = None
    try:
        json.loads('{}\na\nb')
    except json.JSONDecodeError:
        err = sys.exc_info()[1]
    else:
        self.fail('Expected JSONDecodeError')
    s = pickle.dumps(err)
    e = pickle.loads(s)
    self.assertEqual(err.msg, e.msg)
    self.assertEqual(err.doc, e.doc)
    self.assertEqual(err.pos, e.pos)
    self.assertEqual(err.end, e.end)