import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_num_two_qubit_gates_required_invalid():
    with pytest.raises(ValueError, match='(4,4)'):
        cirq.num_cnots_required(np.array([[1]]))