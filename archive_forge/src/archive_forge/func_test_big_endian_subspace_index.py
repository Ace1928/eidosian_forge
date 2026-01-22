import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_big_endian_subspace_index():
    state = np.zeros(shape=(2, 3, 4, 5, 1, 6, 1, 1))
    args = cirq.ApplyUnitaryArgs(state, np.empty_like(state), [1, 3])
    s = slice(None)
    assert args.subspace_index(little_endian_bits_int=1) == (s, 1, s, 0, s, s, s, s)
    assert args.subspace_index(big_endian_bits_int=1) == (s, 0, s, 1, s, s, s, s)