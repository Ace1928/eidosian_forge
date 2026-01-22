import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_cz_act_on_tableau():
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(cirq.CZ, ExampleSimulationState(), qubits=())
    original_tableau = cirq.CliffordTableau(num_qubits=5, initial_state=31)
    state = cirq.CliffordTableauSimulationState(tableau=original_tableau.copy(), qubits=cirq.LineQubit.range(5), prng=np.random.RandomState())
    cirq.act_on(cirq.CZ, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau.stabilizers() == [cirq.DensePauliString('ZIIII', coefficient=-1), cirq.DensePauliString('IZIII', coefficient=-1), cirq.DensePauliString('IIZII', coefficient=-1), cirq.DensePauliString('IIIZI', coefficient=-1), cirq.DensePauliString('IIIIZ', coefficient=-1)]
    assert state.tableau.destabilizers() == [cirq.DensePauliString('XZIII', coefficient=1), cirq.DensePauliString('ZXIII', coefficient=1), cirq.DensePauliString('IIXII', coefficient=1), cirq.DensePauliString('IIIXI', coefficient=1), cirq.DensePauliString('IIIIX', coefficient=1)]
    cirq.act_on(cirq.CZ, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau
    cirq.act_on(cirq.CZ ** 4, state, cirq.LineQubit.range(2), allow_decompose=False)
    assert state.log_of_measurement_results == {}
    assert state.tableau == original_tableau
    foo = sympy.Symbol('foo')
    with pytest.raises(TypeError, match='Failed to act action on state'):
        cirq.act_on(cirq.CZ ** foo, state, cirq.LineQubit.range(2))
    with pytest.raises(TypeError, match='Failed to act action on state'):
        cirq.act_on(cirq.CZ ** 1.5, state, cirq.LineQubit.range(2))