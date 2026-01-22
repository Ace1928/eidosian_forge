import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
@pytest.mark.parametrize('decompose_mode', ['intercept', 'fallback'])
def test_decompose_preserving_structure_forwards_args(decompose_mode):
    a, b = cirq.LineQubit.range(2)
    fc1 = cirq.FrozenCircuit(cirq.SWAP(a, b), cirq.FSimGate(0.1, 0.2).on(a, b))
    cop1_1 = cirq.CircuitOperation(fc1).with_tags('test_tag')
    cop1_2 = cirq.CircuitOperation(fc1).with_qubit_mapping({a: b, b: a})
    fc2 = cirq.FrozenCircuit(cirq.X(a), cop1_1, cop1_2)
    cop2 = cirq.CircuitOperation(fc2)
    circuit = cirq.Circuit(cop2, cirq.measure(a, b, key='m'))

    def keep_func(op: 'cirq.Operation'):
        return not isinstance(op.gate, (cirq.SwapPowGate, cirq.XPowGate))

    def x_to_hzh(op: 'cirq.Operation'):
        if isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1:
            return [cirq.H(*op.qubits), cirq.Z(*op.qubits), cirq.H(*op.qubits)]
    actual = cirq.Circuit(cirq.decompose(circuit, keep=keep_func, intercepting_decomposer=x_to_hzh if decompose_mode == 'intercept' else None, fallback_decomposer=x_to_hzh if decompose_mode == 'fallback' else None, preserve_structure=True))
    fc1_decomp = cirq.FrozenCircuit(cirq.decompose(fc1, keep=keep_func, fallback_decomposer=x_to_hzh))
    expected = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.H(a), cirq.Z(a), cirq.H(a), cirq.CircuitOperation(fc1_decomp).with_tags('test_tag'), cirq.CircuitOperation(fc1_decomp).with_qubit_mapping({a: b, b: a}))), cirq.measure(a, b, key='m'))
    assert actual == expected