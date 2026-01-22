import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
def test_bad_qubit_pairs():
    a, b, c, d, e = cirq.LineQubit.range(5)
    bad_qubit_pairs = [(a, b), (c, d), (e,)]
    with pytest.raises(ValueError):
        cca.strategies.quartic_paired.qubit_pairs_to_qubit_order(bad_qubit_pairs)