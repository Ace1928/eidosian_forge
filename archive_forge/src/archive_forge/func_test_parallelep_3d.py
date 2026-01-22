import pytest
import numpy as np
import cirq
from cirq_pasqal import ThreeDQubit, TwoDQubit
def test_parallelep_3d():
    assert ThreeDQubit.parallelep(1, 2, 2, x0=5, y0=6, z0=7) == [ThreeDQubit(5, 6, 7), ThreeDQubit(5, 7, 7), ThreeDQubit(5, 6, 8), ThreeDQubit(5, 7, 8)]
    assert ThreeDQubit.parallelep(2, 2, 2) == [ThreeDQubit(0, 0, 0), ThreeDQubit(1, 0, 0), ThreeDQubit(0, 1, 0), ThreeDQubit(1, 1, 0), ThreeDQubit(0, 0, 1), ThreeDQubit(1, 0, 1), ThreeDQubit(0, 1, 1), ThreeDQubit(1, 1, 1)]