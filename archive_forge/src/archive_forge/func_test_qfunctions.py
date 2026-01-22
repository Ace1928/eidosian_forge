import math
import pytest
from mpmath import *
def test_qfunctions():
    mp.dps = 15
    assert qp(2, 3, 100).ae('2.7291482267247332183e2391')