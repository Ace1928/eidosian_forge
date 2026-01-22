import unittest
import simplejson as json
def test_prevent_script_breakout(self):
    bad_string = '</script><script>alert("gotcha")</script>'
    self.assertEqual('"\\u003c/script\\u003e\\u003cscript\\u003ealert(\\"gotcha\\")\\u003c/script\\u003e"', self.encoder.encode(bad_string))
    self.assertEqual(bad_string, self.decoder.decode(self.encoder.encode(bad_string)))