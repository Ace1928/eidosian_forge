import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testJonesPolynomial(self):
    L = LaurentPolynomialRing(QQ, 'q')
    q = L.gen()
    data = [('K3_1', ('q^3 + q - 1', -4)), ('K7_2', ('q^7 - q^6 + 2*q^5 - 2*q^4 + 2*q^3 - q^2 + q - 1', -8)), ('K8_3', ('q^8 - q^7 + 2*q^6 - 3*q^5 + 3*q^4 - 3*q^3 + 2*q^2 - q + 1', -4)), ('K8_13', ('-q^8 + 2*q^7 - 3*q^6 + 5*q^5 - 5*q^4 + 5*q^3 - 4*q^2 + 3*q - 1', -3)), ('L6a2', ('-q^6 + q^5 - 2*q^4 + 2*q^3 - 2*q^2 + q - 1', 1)), ('L6a4', ('-q^6 + 3*q^5 - 2*q^4 + 4*q^3 - 2*q^2 + 3*q - 1', -3)), ('L7a3', ('-q^7 + q^6 - 3*q^5 + 2*q^4 - 3*q^3 + 3*q^2 - 2*q + 1', -7)), ('L10n1', ('q^8 - 2*q^7 + 2*q^6 - 4*q^5 + 3*q^4 - 3*q^3 + 2*q^2 - 2*q + 1', -2))]
    for link_name, (poly, exp) in data:
        link = getattr(self, link_name)
        self.assertEqual(link.jones_polynomial(new_convention=False), L(poly) * q ** exp)