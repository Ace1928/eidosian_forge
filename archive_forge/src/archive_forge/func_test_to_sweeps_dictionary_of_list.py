import itertools
import pytest
import sympy
import cirq
def test_to_sweeps_dictionary_of_list():
    with pytest.warns(DeprecationWarning, match='dict_to_product_sweep'):
        assert cirq.study.to_sweeps({'t': [0, 2, 3]}) == cirq.study.to_sweeps([{'t': 0}, {'t': 2}, {'t': 3}])
        assert cirq.study.to_sweeps({'t': [0, 1], 's': [2, 3], 'r': 4}) == cirq.study.to_sweeps([{'t': 0, 's': 2, 'r': 4}, {'t': 0, 's': 3, 'r': 4}, {'t': 1, 's': 2, 'r': 4}, {'t': 1, 's': 3, 'r': 4}])