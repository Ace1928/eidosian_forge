import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testLinkingMatrix(self):
    A = [[0]]
    B = [[0, -1], [-1, 0]]
    B2 = [[0, 1], [1, 0]]
    C = [[0, -3], [-3, 0]]
    C2 = [[0, 3], [3, 0]]
    D = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    E = [[0, 0], [0, 0]]
    for k in self.knots:
        self.assertEqual(k.linking_matrix(), A)
    self.assertIn(self.L2a1.linking_matrix(), [B, B2])
    self.assertIn(self.L6a2.linking_matrix(), [C, C2])
    self.assertEqual(self.Borr.linking_matrix(), D)
    self.assertEqual(self.L6a4.linking_matrix(), D)
    self.assertEqual(self.L7a3.linking_matrix(), E)