import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('terms, terms_expected', (({}, {}), ({'key': 1}, {'key': -1}), ({'1': 10, '2': -20}, {'1': -10, '2': 20})))
def test_vector_negation(terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual = -linear_dict
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual