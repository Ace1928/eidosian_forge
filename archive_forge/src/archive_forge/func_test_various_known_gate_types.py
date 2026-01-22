import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_various_known_gate_types():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.X(a), cirq.X(a) ** 0.25, cirq.X(a) ** (-0.5), cirq.Z(a), cirq.Z(a) ** 0.5, cirq.Y(a), cirq.Y(a) ** (-0.25), cirq.Y(a) ** sympy.Symbol('t'), cirq.H(a), cirq.measure(a), cirq.measure(a, b, key='not-relevant'), cirq.SWAP(a, b), cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.CZ(a, b))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[\n            ["X"],\n            ["X^¼"],\n            ["X^-½"],\n            ["Z"],\n            ["Z^½"],\n            ["Y"],\n            ["Y^-¼"],\n            ["Y^t"],\n            ["H"],\n            ["Measure"],\n            ["Measure","Measure"],\n            ["Swap","Swap"],\n            ["•","X"],\n            ["X","•"],\n            ["•","Z"]]}\n    ', escape_url=False)