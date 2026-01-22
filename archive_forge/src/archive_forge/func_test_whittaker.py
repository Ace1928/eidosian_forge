import math
import pytest
from mpmath import *
def test_whittaker():
    mp.dps = 15
    assert whitm(2, 3, 4).ae(49.75374558902524)
    assert whitw(2, 3, 4).ae(14.111656223052933)