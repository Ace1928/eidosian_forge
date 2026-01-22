import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_moment_text_diagram():
    a, b, c, d = cirq.GridQubit.rect(2, 2)
    m = cirq.Moment(cirq.CZ(a, b), cirq.CNOT(c, d))
    assert str(m).strip() == '\n  ╷ 0 1\n╶─┼─────\n0 │ @─@\n  │\n1 │ @─X\n  │\n    '.strip()
    m = cirq.Moment(cirq.CZ(a, b), cirq.CNOT(c, d))
    cirq.testing.assert_has_diagram(m, '\n   ╷ None 0 1\n╶──┼──────────\naa │\n   │\n0  │      @─@\n   │\n1  │      @─X\n   │\n        ', extra_qubits=[cirq.NamedQubit('aa')])
    m = cirq.Moment(cirq.S(c), cirq.ISWAP(a, d))
    cirq.testing.assert_has_diagram(m, '\n  ╷ 0     1\n╶─┼─────────────\n0 │ iSwap─┐\n  │       │\n1 │ S     iSwap\n  │\n    ')
    m = cirq.Moment(cirq.S(c) ** 0.1, cirq.ISWAP(a, d) ** 0.5)
    cirq.testing.assert_has_diagram(m, '\n  ╷ 0         1\n╶─┼─────────────────\n0 │ iSwap^0.5─┐\n  │           │\n1 │ Z^0.05    iSwap\n  │\n    ')
    a, b, c = cirq.LineQubit.range(3)
    m = cirq.Moment(cirq.X(a), cirq.SWAP(b, c))
    cirq.testing.assert_has_diagram(m, '\n  ╷ a b c\n╶─┼───────\n0 │ X\n  │\n1 │   ×─┐\n  │     │\n2 │     ×\n  │\n    ', xy_breakdown_func=lambda q: ('abc'[q.x], q.x))

    class EmptyGate(cirq.testing.SingleQubitGate):

        def __str__(self):
            return 'Empty'
    m = cirq.Moment(EmptyGate().on(a))
    cirq.testing.assert_has_diagram(m, '\n  ╷ 0\n╶─┼───────\n0 │ Empty\n  │\n    ')