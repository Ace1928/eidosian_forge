import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_controls():
    a, b = cirq.LineQubit.range(2)
    assert_url_to_circuit_returns('{"cols":[["•","X"]]}', cirq.Circuit(cirq.X(b).controlled_by(a)))
    assert_url_to_circuit_returns('{"cols":[["◦","X"]]}', cirq.Circuit(cirq.X(a), cirq.X(b).controlled_by(a), cirq.X(a)))
    assert_url_to_circuit_returns('{"cols":[["⊕","X"]]}', cirq.Circuit(cirq.Y(a) ** 0.5, cirq.X(b).controlled_by(a), cirq.Y(a) ** (-0.5)), output_amplitudes_from_quirk=[{'r': 0.5, 'i': 0}, {'r': -0.5, 'i': 0}, {'r': 0.5, 'i': 0}, {'r': 0.5, 'i': 0}])
    assert_url_to_circuit_returns('{"cols":[["⊖","X"]]}', cirq.Circuit(cirq.Y(a) ** (-0.5), cirq.X(b).controlled_by(a), cirq.Y(a) ** (+0.5)), output_amplitudes_from_quirk=[{'r': 0.5, 'i': 0}, {'r': 0.5, 'i': 0}, {'r': 0.5, 'i': 0}, {'r': -0.5, 'i': 0}])
    assert_url_to_circuit_returns('{"cols":[["⊗","X"]]}', cirq.Circuit(cirq.X(a) ** (-0.5), cirq.X(b).controlled_by(a), cirq.X(a) ** (+0.5)), output_amplitudes_from_quirk=[{'r': 0.5, 'i': 0}, {'r': 0, 'i': -0.5}, {'r': 0.5, 'i': 0}, {'r': 0, 'i': 0.5}])
    assert_url_to_circuit_returns('{"cols":[["(/)","X"]]}', cirq.Circuit(cirq.X(a) ** (+0.5), cirq.X(b).controlled_by(a), cirq.X(a) ** (-0.5)), output_amplitudes_from_quirk=[{'r': 0.5, 'i': 0}, {'r': 0, 'i': 0.5}, {'r': 0.5, 'i': 0}, {'r': 0, 'i': -0.5}])
    qs = cirq.LineQubit.range(8)
    assert_url_to_circuit_returns('{"cols":[["X","•","◦","⊕","⊖","⊗","(/)","Z"]]}', cirq.Circuit(cirq.X(qs[2]), cirq.Y(qs[3]) ** 0.5, cirq.Y(qs[4]) ** (-0.5), cirq.X(qs[5]) ** (-0.5), cirq.X(qs[6]) ** 0.5, cirq.X(qs[0]).controlled_by(*qs[1:7]), cirq.Z(qs[7]).controlled_by(*qs[1:7]), cirq.X(qs[6]) ** (-0.5), cirq.X(qs[5]) ** 0.5, cirq.Y(qs[4]) ** 0.5, cirq.Y(qs[3]) ** (-0.5), cirq.X(qs[2])))