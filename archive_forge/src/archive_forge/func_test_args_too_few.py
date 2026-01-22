import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_args_too_few(self):
    with pytest.raises(TypeError):
        Affine(1, 2)