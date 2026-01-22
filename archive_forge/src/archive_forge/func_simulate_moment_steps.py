import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def simulate_moment_steps(self, circuit: 'cirq.AbstractCircuit', param_resolver: 'cirq.ParamResolverOrSimilarType'=None, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TStepResult]:
    """Returns an iterator of StepResults for each moment simulated.

        If the circuit being simulated is empty, a single step result should
        be returned with the state being set to the initial state.

        Args:
            circuit: The Circuit to simulate.
            param_resolver: A ParamResolver for determining values of Symbols.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. This can be
                either a raw state or a `TSimulationState`. The form of the
                raw state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            Iterator that steps through the simulation, simulating each
            moment and returning a StepResult for each moment.
        """
    param_resolver = study.ParamResolver(param_resolver)
    resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
    check_all_resolved(resolved_circuit)
    actual_initial_state = 0 if initial_state is None else initial_state
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
    return self._base_iterator(resolved_circuit, qubits, actual_initial_state)