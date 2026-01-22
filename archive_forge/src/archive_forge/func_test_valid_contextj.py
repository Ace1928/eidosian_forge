import unittest
import idna
def test_valid_contextj(self):
    zwnj = '\u200c'
    zwj = '\u200d'
    virama = '्'
    latin = 'a'
    self.assertFalse(idna.valid_contextj(zwnj, 0))
    self.assertFalse(idna.valid_contextj(latin + zwnj, 1))
    self.assertTrue(idna.valid_contextj(virama + zwnj, 1))
    self.assertFalse(idna.valid_contextj(zwj, 0))
    self.assertFalse(idna.valid_contextj(latin + zwj, 1))
    self.assertTrue(idna.valid_contextj(virama + zwj, 1))