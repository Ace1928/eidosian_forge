import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_imul_number():
    t = Affine(1, 2, 3, 4, 5, 6)
    try:
        t *= 2.0
    except TypeError:
        assert True