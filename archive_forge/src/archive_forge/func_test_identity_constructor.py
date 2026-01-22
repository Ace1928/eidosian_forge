import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_identity_constructor(self):
    ident = Affine.identity()
    assert isinstance(ident, Affine)
    assert tuple(ident) == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert ident.is_identity