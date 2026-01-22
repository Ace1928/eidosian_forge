import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
def simulate_sweep_iter(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TSimulationTrialResult]:
    """Simulates the supplied Circuit.

        This particular implementation overrides the base implementation such
        that an unparameterized prefix circuit is simulated and fed into the
        parameterized suffix circuit.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. This can be
                either a raw state or an `SimulationStateBase`. The form of the
                raw state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """

    def sweep_prefixable(op: 'cirq.Operation'):
        return self._can_be_in_run_prefix(op) and (not protocols.is_parameterized(op))
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(program.all_qubits())
    initial_state = 0 if initial_state is None else initial_state
    sim_state = self._create_simulation_state(initial_state, qubits)
    prefix, suffix = split_into_matching_protocol_then_general(program, sweep_prefixable) if self._can_be_in_run_prefix(self.noise) else (program[0:0], program)
    step_result = None
    for step_result in self._core_iterator(circuit=prefix, sim_state=sim_state):
        pass
    sim_state = step_result._sim_state
    yield from super().simulate_sweep_iter(suffix, params, qubit_order, sim_state)