import cirq
import numpy as np
import pytest
def test_matrix_mixture_equality():
    dp_pt1 = cirq.depolarize(0.1)
    dp_pt2 = cirq.depolarize(0.2)
    mm_a1 = cirq.MixedUnitaryChannel.from_mixture(dp_pt1, key='a')
    mm_a2 = cirq.MixedUnitaryChannel.from_mixture(dp_pt2, key='a')
    mm_b1 = cirq.MixedUnitaryChannel.from_mixture(dp_pt1, key='b')
    assert mm_a1 != dp_pt1
    assert mm_a1 != mm_a2
    assert mm_a1 != mm_b1
    assert mm_a2 != mm_b1
    mix = [(0.5, np.array([[1, 0], [0, 1]])), (0.5, np.array([[0, 1], [1, 0]]))]
    half_flip = cirq.MixedUnitaryChannel(mix)
    mix_inv = list(reversed(mix))
    half_flip_inv = cirq.MixedUnitaryChannel(mix_inv)
    assert half_flip != half_flip_inv