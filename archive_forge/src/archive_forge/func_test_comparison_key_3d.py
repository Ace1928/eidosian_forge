import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_comparison_key_3d():
    assert ThreeDQubit(3, 4, 5)._comparison_key() == (5, 4, 3)
    coords = (np.cos(np.pi / 2), np.sin(np.pi / 2), 0)
    assert ThreeDQubit(*coords) == ThreeDQubit(0, 1, 0)