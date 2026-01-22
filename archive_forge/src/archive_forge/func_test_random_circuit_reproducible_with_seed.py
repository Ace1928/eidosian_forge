from typing import Optional, Dict, Sequence, Union, cast
import random
import numpy as np
import pytest
import cirq
import cirq.testing
@pytest.mark.parametrize('seed', [random.randint(0, 2 ** 32) for _ in range(10)])
def test_random_circuit_reproducible_with_seed(seed):
    wrappers = (lambda s: s, np.random.RandomState)
    circuits = [cirq.testing.random_circuit(qubits=10, n_moments=10, op_density=0.7, random_state=wrapper(seed)) for wrapper in wrappers for _ in range(2)]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*circuits)