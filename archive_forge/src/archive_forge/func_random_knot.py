import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def random_knot(self):
    random_index = randrange(0, len(self.knots))
    return self.knots[random_index]