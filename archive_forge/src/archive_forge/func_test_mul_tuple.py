import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_mul_tuple():
    t = Affine(1, 2, 3, 4, 5, 6)
    t * (2.0, 2.0)