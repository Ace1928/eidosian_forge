import operator as op
from .. import units as pq
from ..dimensionality import Dimensionality
from .common import TestCase
def test_dimensionality_str(self):
    self.assertEqual(str(meter), meter_str)
    self.assertEqual(joule.string, joule_str)
    self.assertEqual(joule.unicode, joule_uni)
    self.assertEqual(joule.latex, joule_tex)
    self.assertEqual(joule.html, joule_htm)
    self.assertEqual(Joule.string, 'J')