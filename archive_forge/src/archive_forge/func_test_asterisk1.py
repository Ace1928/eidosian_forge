import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from . import util
def test_asterisk1(self):
    foo = getattr(self.module, 'foo1')
    assert_equal(foo(), b'123456789A12')