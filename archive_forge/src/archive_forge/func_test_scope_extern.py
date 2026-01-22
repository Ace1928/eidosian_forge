import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_scope_extern():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(cirq.measure(q, key='a'), cirq.X(q).with_classical_controls('b'))
    middle = cirq.Circuit(cirq.measure(q, key=cirq.MeasurementKey('b')), cirq.CircuitOperation(inner.freeze(), repetitions=2))
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)]
    assert internal_control_keys == ['0:b', '0:b', '1:b', '1:b']
    assert not cirq.control_keys(outer_subcircuit)
    assert not cirq.control_keys(circuit)
    cirq.testing.assert_has_diagram(cirq.Circuit(outer_subcircuit), "\n      [           [ 0: ───M('a')───X─── ]             ]\n      [ 0: ───M───[                ║    ]──────────── ]\n0: ───[       ║   [ b: ════════════^═══ ](loops=2)    ]────────────\n      [       ║   ║                                   ]\n      [ b: ═══@═══╩══════════════════════════════════ ](loops=2)\n", use_unicode_characters=True)
    cirq.testing.assert_has_diagram(circuit, "\n0: ─────M───M('0:0:a')───X───M('0:1:a')───X───M───M('1:0:a')───X───M('1:1:a')───X───\n        ║                ║                ║   ║                ║                ║\n0:b: ═══@════════════════^════════════════^═══╬════════════════╬════════════════╬═══\n                                              ║                ║                ║\n1:b: ═════════════════════════════════════════@════════════════^════════════════^═══\n", use_unicode_characters=True)
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))