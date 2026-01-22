import pickle
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('dtype', (np.int8, np.int16, np.int32, np.int64, int))
def test_addition_subtraction_numpy_array(dtype):
    assert cirq.GridQubit(1, 2) + np.array([1, 2], dtype=dtype) == cirq.GridQubit(2, 4)
    assert cirq.GridQubit(1, 2) + np.array([0, 0], dtype=dtype) == cirq.GridQubit(1, 2)
    assert cirq.GridQubit(1, 2) + np.array([-1, 0], dtype=dtype) == cirq.GridQubit(0, 2)
    assert cirq.GridQubit(1, 2) - np.array([1, 2], dtype=dtype) == cirq.GridQubit(0, 0)
    assert cirq.GridQubit(1, 2) - np.array([0, 0], dtype=dtype) == cirq.GridQubit(1, 2)
    assert cirq.GridQid(1, 2, dimension=3) - np.array([-1, 0], dtype=dtype) == cirq.GridQid(2, 2, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) + np.array([1, 2], dtype=dtype) == cirq.GridQid(2, 4, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) + np.array([0, 0], dtype=dtype) == cirq.GridQid(1, 2, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) + np.array([-1, 0], dtype=dtype) == cirq.GridQid(0, 2, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) - np.array([1, 2], dtype=dtype) == cirq.GridQid(0, 0, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) - np.array([0, 0], dtype=dtype) == cirq.GridQid(1, 2, dimension=3)
    assert cirq.GridQid(1, 2, dimension=3) - np.array([-1, 0], dtype=dtype) == cirq.GridQid(2, 2, dimension=3)