import unittest
import simplejson as json
def test_for_json_encodes_object_nested_in_dict(self):
    self.assertRoundTrip({'hooray': ForJson()}, {'hooray': ForJson().for_json()})