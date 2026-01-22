import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_names_are_unique(self):
    assert len(set(self.numeric_types)) == len(self.numeric_types)
    names = [t.__name__ for t in self.numeric_types]
    assert len(set(names)) == len(names)