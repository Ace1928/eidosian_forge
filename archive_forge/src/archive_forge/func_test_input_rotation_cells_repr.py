import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_input_rotation_cells_repr():
    circuit = quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["•","X^(-A/2^n)","inputA2"]]}')
    op = circuit[0].operations[0]
    cirq.testing.assert_equivalent_repr(op)