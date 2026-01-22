from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_sort_keys(self):
    for num_keys in range(2, 32):
        p = dict(((str(x), x) for x in range(num_keys)))
        sio = StringIO()
        json.dump(p, sio, sort_keys=True)
        self.assertEqual(sio.getvalue(), json.dumps(p, sort_keys=True))
        self.assertEqual(json.loads(sio.getvalue()), p)