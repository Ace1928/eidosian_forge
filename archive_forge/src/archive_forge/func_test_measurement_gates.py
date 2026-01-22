import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_measurement_gates():
    a, b, c = cirq.LineQubit.range(3)
    assert_url_to_circuit_returns('{"cols":[["Measure","Measure"],["Measure","Measure"]]}', cirq.Circuit(cirq.measure(a, key='row=0,col=0'), cirq.measure(b, key='row=1,col=0'), cirq.measure(a, key='row=0,col=1'), cirq.measure(b, key='row=1,col=1')))
    assert_url_to_circuit_returns('{"cols":[["XDetector","YDetector","ZDetector"]]}', cirq.Circuit(cirq.X(b) ** (-0.5), cirq.Y(a) ** 0.5, cirq.Moment([cirq.measure(a, key='row=0,col=0'), cirq.measure(b, key='row=1,col=0'), cirq.measure(c, key='row=2,col=0')]), cirq.Y(a) ** (-0.5), cirq.X(b) ** 0.5))