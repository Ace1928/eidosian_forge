import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_slice_last_row(self):
    t = Affine(1, 2, 3, 4, 5, 6)
    assert t[-3:] == (0, 0, 1)