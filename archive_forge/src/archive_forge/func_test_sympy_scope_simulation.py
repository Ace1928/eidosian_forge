import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_sympy_scope_simulation():
    q0, q1, q2, q3, q_ignored, q_result = cirq.LineQubit.range(6)
    condition = sympy_parser.parse_expr('a & b | c & d')
    for i in range(32):
        bits = cirq.big_endian_int_to_bits(i, bit_count=5)
        inner = cirq.Circuit(cirq.X(q0) ** bits[0], cirq.measure(q0, key='a'), cirq.X(q_result).with_classical_controls(condition), cirq.measure(q_result, key='m_result'))
        middle = cirq.Circuit(cirq.X(q1) ** bits[1], cirq.measure(q1, key='b'), cirq.X(q_ignored) ** bits[4], cirq.measure(q_ignored, key=cirq.MeasurementKey('c', ('0',))), cirq.CircuitOperation(inner.freeze(), repetition_ids=['0']))
        circuit = cirq.Circuit(cirq.X(q2) ** bits[2], cirq.measure(q2, key='c'), cirq.X(q3) ** bits[3], cirq.measure(q3, key='d'), cirq.CircuitOperation(middle.freeze(), repetition_ids=['0']))
        result = cirq.CliffordSimulator().run(circuit)
        assert result.measurements['0:0:m_result'][0][0] == (bits[0] and bits[1] or (bits[2] and bits[3]))