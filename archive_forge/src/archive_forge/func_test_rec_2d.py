import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_rec_2d():
    assert TwoDQubit.rect(1, 2, x0=5, y0=6) == [TwoDQubit(5, 6), TwoDQubit(5, 7)]
    assert TwoDQubit.rect(2, 2) == [TwoDQubit(0, 0), TwoDQubit(1, 0), TwoDQubit(0, 1), TwoDQubit(1, 1)]