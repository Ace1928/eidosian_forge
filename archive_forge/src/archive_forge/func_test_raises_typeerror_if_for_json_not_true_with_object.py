import unittest
import simplejson as json
def test_raises_typeerror_if_for_json_not_true_with_object(self):
    self.assertRaises(TypeError, json.dumps, ForJson())
    self.assertRaises(TypeError, json.dumps, ForJson(), for_json=False)