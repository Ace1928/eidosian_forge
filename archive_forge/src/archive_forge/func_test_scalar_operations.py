import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_scalar_operations():
    assert_url_to_circuit_returns('{"cols":[["…"]]}', cirq.Circuit())
    assert_url_to_circuit_returns('{"cols":[["NeGate"]]}', cirq.Circuit(cirq.global_phase_operation(-1)))
    assert_url_to_circuit_returns('{"cols":[["i"]]}', cirq.Circuit(cirq.global_phase_operation(1j)))
    assert_url_to_circuit_returns('{"cols":[["-i"]]}', cirq.Circuit(cirq.global_phase_operation(-1j)))
    assert_url_to_circuit_returns('{"cols":[["√i"]]}', cirq.Circuit(cirq.global_phase_operation(1j ** 0.5)))
    assert_url_to_circuit_returns('{"cols":[["√-i"]]}', cirq.Circuit(cirq.global_phase_operation(1j ** (-0.5))))