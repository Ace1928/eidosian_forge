import unittest
import simplejson as json
def test_for_json_encodes_stand_alone_object(self):
    self.assertRoundTrip(ForJson(), ForJson().for_json())