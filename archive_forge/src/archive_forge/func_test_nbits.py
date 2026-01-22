from Cryptodome.Util.py3compat import *
import unittest
def test_nbits(self):
    c = Counter.new(nbits=128)
    self.assertRaises(ValueError, Counter.new, 129)