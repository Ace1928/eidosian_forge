import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('terms, valid_vectors', (({'X': 2}, 'X'), ({'X': 2, 'Y': -2}, ('X', 'Y', 'Z'))))
def test_valid_vectors_are_accepted(terms, valid_vectors):
    linear_dict = cirq.LinearDict(terms, validator=lambda v: v in valid_vectors)
    original_dict = linear_dict.copy()
    delta_dict = cirq.LinearDict.fromkeys(valid_vectors, 1)
    linear_dict += cirq.LinearDict.fromkeys(valid_vectors, 1)
    assert linear_dict == original_dict + delta_dict
    for vector in valid_vectors:
        linear_dict[vector] += 1
    assert linear_dict == original_dict + 2 * delta_dict
    linear_dict.update(cirq.LinearDict.fromkeys(valid_vectors, 1))
    assert linear_dict == delta_dict