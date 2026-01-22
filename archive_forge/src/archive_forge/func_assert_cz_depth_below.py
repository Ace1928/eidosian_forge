import cmath
import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.analytical_decompositions.two_qubit_to_cz import (
from cirq.testing import random_two_qubit_circuit_with_czs
def assert_cz_depth_below(operations, threshold, must_be_full):
    total_cz = 0
    for op in operations:
        assert len(op.qubits) <= 2
        if len(op.qubits) == 2:
            assert isinstance(op.gate, cirq.CZPowGate)
            e = value.canonicalize_half_turns(op.gate.exponent)
            if must_be_full:
                assert e == 1
            total_cz += abs(e)
    assert total_cz <= threshold