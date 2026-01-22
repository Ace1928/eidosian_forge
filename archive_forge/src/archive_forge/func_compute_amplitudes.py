import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def compute_amplitudes(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], param_resolver: 'cirq.ParamResolverOrSimilarType'=None, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Sequence[complex]:
    """Computes the desired amplitudes.

        The initial state is assumed to be the all zeros state.

        Args:
            program: The circuit to simulate.
            bitstrings: The bitstrings whose amplitudes are desired, input
                as an integer array where each integer is formed from measured
                qubit values according to `qubit_order` from most to least
                significant qubit, i.e. in big-endian ordering. If inputting
                a binary literal add the prefix 0b or 0B.
                For example: 0010 can be input as 0b0010, 0B0010, 2, 0x2, etc.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.

        Returns:
            List of amplitudes.
        """
    return self.compute_amplitudes_sweep(program, bitstrings, study.ParamResolver(param_resolver), qubit_order)[0]