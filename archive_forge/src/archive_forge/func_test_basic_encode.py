import unittest
import simplejson as json
def test_basic_encode(self):
    self.assertEqual('"\\u0026"', self.encoder.encode('&'))
    self.assertEqual('"\\u003c"', self.encoder.encode('<'))
    self.assertEqual('"\\u003e"', self.encoder.encode('>'))
    self.assertEqual('"\\u2028"', self.encoder.encode(u'\u2028'))