import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_warns_byteorder(self):
    other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
    with pytest.deprecated_call(match='non-native byteorder is not'):
        random.randint(0, 200, size=10, dtype=other_byteord_dt)