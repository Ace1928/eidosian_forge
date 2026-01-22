import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('pauli', (cirq.X, cirq.Y, cirq.Z))
def test_list_op_constructor_matches_mapping(pauli):
    q0, = _make_qubits(1)
    op = pauli.on(q0)
    assert cirq.PauliString([op]) == cirq.PauliString({q0: pauli})