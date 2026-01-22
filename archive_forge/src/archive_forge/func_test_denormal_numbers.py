import pytest
from numpy import (
from numpy.testing import (
def test_denormal_numbers(self):
    for ftype in sctypes['float']:
        stop = nextafter(ftype(0), ftype(1)) * 5
        assert_(any(linspace(0, stop, 10, endpoint=False, dtype=ftype)))