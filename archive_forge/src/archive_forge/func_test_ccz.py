import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_ccz():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CCZ(a, b, c))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["•","•","Z"]]}\n    ', escape_url=False)
    circuit = cirq.Circuit(cirq.CCZ(a, b, c) ** sympy.Symbol('t'))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["•","•","Z^t"]]}\n    ', escape_url=False)
    circuit = cirq.Circuit(cirq.CCZ(a, b, c) ** 0.01)
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[\n            ["UNKNOWN","UNKNOWN","UNKNOWN"]\n        ]}\n    ', escape_url=False, prefer_unknown_gate_to_failure=True)
    circuit = cirq.Circuit(cirq.CCZ(a, b, c) ** 0.5, cirq.H(d))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[\n            ["•","•","Z^½"],[1,1,1,"H"]]}\n    ', escape_url=False)