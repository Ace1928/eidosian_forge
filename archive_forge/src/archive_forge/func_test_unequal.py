import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('terms_1, terms_2', (({}, {'a': 1}), ({'X': 1e-12}, {'X': 0}), ({'X': 0.0}, {'Y': 0.1}), ({'X': 1}, {'X': 1, 'Z': 1e-12})))
def test_unequal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert linear_dict_1 != linear_dict_2
    assert linear_dict_2 != linear_dict_1
    assert not linear_dict_1 == linear_dict_2
    assert not linear_dict_2 == linear_dict_1