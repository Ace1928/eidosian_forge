import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('scalar, terms, terms_expected', ((2, {}, {}), (2, {'X': 1, 'Y': -2}, {'X': 2, 'Y': -4}), (0, {'abc': 10, 'def': 20}, {}), (1j, {'X': 4j}, {'X': -4}), (-1, {'a': 10, 'b': -20}, {'a': -10, 'b': 20})))
def test_scalar_multiplication(scalar, terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual_1 = scalar * linear_dict
    actual_2 = linear_dict * scalar
    expected = cirq.LinearDict(terms_expected)
    assert actual_1 == expected
    assert actual_2 == expected
    assert actual_1 == actual_2