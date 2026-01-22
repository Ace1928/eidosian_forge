import unittest
import idna.compat
def testToASCII(self):
    self.assertEqual(idna.compat.ToASCII('テスト.xn--zckzah'), b'xn--zckzah.xn--zckzah')