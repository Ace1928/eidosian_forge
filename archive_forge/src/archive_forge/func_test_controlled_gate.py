import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
def test_controlled_gate():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.ControlledGate(cirq.ControlledGate(cirq.CZ)).on(a, d, c, b))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["•","Z","•", "•"]]}\n    ', escape_url=False)
    circuit = cirq.Circuit(cirq.ControlledGate(cirq.X).on(a, b), cirq.ControlledGate(cirq.Z).on(c, d))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["•","X"],[1,1,"•", "Z"]]}\n    ', escape_url=False)
    circuit = cirq.Circuit(cirq.ControlledGate(MysteryGate()).on(a, b))
    assert_links_to(circuit, '\n        http://algassert.com/quirk#circuit={"cols":[["UNKNOWN","UNKNOWN"]]}\n    ', escape_url=False, prefer_unknown_gate_to_failure=True)