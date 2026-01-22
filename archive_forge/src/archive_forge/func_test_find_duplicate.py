import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_find_duplicate():
    l1 = [1, 2, 3, 4, 5, 6]
    assert_(np.rec.find_duplicate(l1) == [])
    l2 = [1, 2, 1, 4, 5, 6]
    assert_(np.rec.find_duplicate(l2) == [1])
    l3 = [1, 2, 1, 4, 1, 6, 2, 3]
    assert_(np.rec.find_duplicate(l3) == [1, 2])
    l3 = [2, 2, 1, 4, 1, 6, 2, 3]
    assert_(np.rec.find_duplicate(l3) == [2, 1])