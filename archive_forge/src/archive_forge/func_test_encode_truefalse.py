from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_encode_truefalse(self):
    self.assertEqual(json.dumps({True: False, False: True}, sort_keys=True), '{"false": true, "true": false}')
    self.assertEqual(json.dumps({2: 3.0, 4.0: long_type(5), False: 1, long_type(6): True, '7': 0}, sort_keys=True), '{"2": 3.0, "4.0": 5, "6": true, "7": 0, "false": 1}')