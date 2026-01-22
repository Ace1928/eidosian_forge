import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_vector_input_not_unitary():
    with pytest.raises(ValueError, match='must correspond to'):
        cirq.kak_vector(np.zeros((4, 4)))