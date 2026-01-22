import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_nonwriteable_setfield(self):
    r = np.rec.array([(0,), (1,)], dtype=[('f', 'i4')])
    r.flags.writeable = False
    with assert_raises(ValueError):
        r.f = [2, 3]
    with assert_raises(ValueError):
        r.setfield([2, 3], *r.dtype.fields['f'])