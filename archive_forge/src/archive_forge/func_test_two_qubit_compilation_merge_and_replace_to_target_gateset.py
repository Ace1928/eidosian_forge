from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def test_two_qubit_compilation_merge_and_replace_to_target_gateset():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.Moment(cirq.Z(q[1]), cirq.X(q[0])), cirq.Moment(cirq.CZ(*q).with_tags('no_compile')), cirq.Moment(cirq.Z.on_each(*q)), cirq.Moment(cirq.X(q[0])), cirq.Moment(cirq.CZ(*q)), cirq.Moment(cirq.Z.on_each(*q)), cirq.Moment(cirq.X(q[0])))
    cirq.testing.assert_has_diagram(c_orig, "\n0: ───X───@['no_compile']───Z───X───@───Z───X───\n          │                         │\n1: ───Z───@─────────────────Z───────@───Z───────\n")
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=ExampleCXTargetGateset(), context=cirq.TransformerContext(tags_to_ignore=('no_compile',)))
    cirq.testing.assert_has_diagram(c_new, "\n0: ───X───@['no_compile']───X───@───Y───@───Z───\n          │                     │       │\n1: ───Z───@─────────────────X───X───Y───X───Z───\n")