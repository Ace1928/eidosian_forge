import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_single_qubit_gate_phased_xz():
    q = cirq.LineQubit(0)
    gate = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=0.3, z_exponent=0.4)
    circuit = cirq.Circuit(gate(q))
    compiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    ops = list(compiled_circuit.all_operations())
    assert len(ops) == 1
    assert ops[0].gate == gate