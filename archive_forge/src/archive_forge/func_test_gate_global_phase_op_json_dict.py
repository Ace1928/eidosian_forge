import numpy as np
import pytest
import sympy
import cirq
def test_gate_global_phase_op_json_dict():
    assert cirq.GlobalPhaseGate(-1j)._json_dict_() == {'coefficient': -1j}