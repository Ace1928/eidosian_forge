import numpy as np
import pytest
from pyquil import Program
from pyquil.simulation.tools import program_unitary
from cirq import Circuit, LineQubit
from cirq_rigetti.quil_input import (
from cirq.ops import (
def test_unsupported_quil_instruction():
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil('NOP')
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil('PRAGMA ADD-KRAUS X 0 "(0.0 1.0 1.0 0.0)"')
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil('RESET')
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil(QUIL_PROGRAM_WITH_PARAMETERIZED_DEFGATE)