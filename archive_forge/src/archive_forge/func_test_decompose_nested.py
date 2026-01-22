import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_decompose_nested():
    a, b, c, d = cirq.LineQubit.range(4)
    exp1 = sympy.Symbol('exp1')
    exp_half = sympy.Symbol('exp_half')
    exp_one = sympy.Symbol('exp_one')
    exp_two = sympy.Symbol('exp_two')
    circuit1 = cirq.FrozenCircuit(cirq.X(a) ** exp1, cirq.measure(a, key='m1'))
    op1 = cirq.CircuitOperation(circuit1)
    circuit2 = cirq.FrozenCircuit(op1.with_qubits(a).with_measurement_key_mapping({'m1': 'ma'}), op1.with_qubits(b).with_measurement_key_mapping({'m1': 'mb'}), op1.with_qubits(c).with_measurement_key_mapping({'m1': 'mc'}), op1.with_qubits(d).with_measurement_key_mapping({'m1': 'md'}))
    op2 = cirq.CircuitOperation(circuit2)
    circuit3 = cirq.FrozenCircuit(op2.with_params({exp1: exp_half}), op2.with_params({exp1: exp_one}).with_measurement_key_mapping({'ma': 'ma1'}).with_measurement_key_mapping({'mb': 'mb1'}).with_measurement_key_mapping({'mc': 'mc1'}).with_measurement_key_mapping({'md': 'md1'}), op2.with_params({exp1: exp_two}).with_measurement_key_mapping({'ma': 'ma2'}).with_measurement_key_mapping({'mb': 'mb2'}).with_measurement_key_mapping({'mc': 'mc2'}).with_measurement_key_mapping({'md': 'md2'}))
    op3 = cirq.CircuitOperation(circuit3)
    final_op = op3.with_params({exp_half: 0.5, exp_one: 1.0, exp_two: 2.0})
    expected_circuit1 = cirq.Circuit(op2.with_params({exp1: 0.5, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0}), op2.with_params({exp1: 1.0, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0}).with_measurement_key_mapping({'ma': 'ma1'}).with_measurement_key_mapping({'mb': 'mb1'}).with_measurement_key_mapping({'mc': 'mc1'}).with_measurement_key_mapping({'md': 'md1'}), op2.with_params({exp1: 2.0, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0}).with_measurement_key_mapping({'ma': 'ma2'}).with_measurement_key_mapping({'mb': 'mb2'}).with_measurement_key_mapping({'mc': 'mc2'}).with_measurement_key_mapping({'md': 'md2'}))
    result_ops1 = cirq.decompose_once(final_op)
    assert cirq.Circuit(result_ops1) == expected_circuit1
    expected_circuit = cirq.Circuit(cirq.X(a) ** 0.5, cirq.measure(a, key='ma'), cirq.X(b) ** 0.5, cirq.measure(b, key='mb'), cirq.X(c) ** 0.5, cirq.measure(c, key='mc'), cirq.X(d) ** 0.5, cirq.measure(d, key='md'), cirq.X(a) ** 1.0, cirq.measure(a, key='ma1'), cirq.X(b) ** 1.0, cirq.measure(b, key='mb1'), cirq.X(c) ** 1.0, cirq.measure(c, key='mc1'), cirq.X(d) ** 1.0, cirq.measure(d, key='md1'), cirq.X(a) ** 2.0, cirq.measure(a, key='ma2'), cirq.X(b) ** 2.0, cirq.measure(b, key='mb2'), cirq.X(c) ** 2.0, cirq.measure(c, key='mc2'), cirq.X(d) ** 2.0, cirq.measure(d, key='md2'))
    assert cirq.Circuit(cirq.decompose(final_op)) == expected_circuit
    assert final_op.mapped_circuit(deep=True) == expected_circuit