import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_bad_type_world(self):
    """wrong type, i.e don't use readlines()"""
    with pytest.raises(TypeError):
        affine.loadsw(['1.0', '0.0', '0.0', '1.0', '0.0', '0.0'])