import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_method_array2(self):
    r = np.rec.array([(1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'), (5, 55, 'ex'), (6, 66, 'f'), (7, 77, 'g')], formats='u1,f4,a1')
    assert_equal(r[1].item(), (2, 22.0, b'b'))