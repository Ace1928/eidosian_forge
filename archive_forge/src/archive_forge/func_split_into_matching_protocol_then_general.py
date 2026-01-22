import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def split_into_matching_protocol_then_general(circuit: 'cirq.AbstractCircuit', predicate: Callable[['cirq.Operation'], bool]) -> Tuple['cirq.AbstractCircuit', 'cirq.AbstractCircuit']:
    """Splits the circuit into a matching prefix and non-matching suffix.

    The splitting happens in a per-qubit fashion. A non-matching operation on
    qubit A will cause later operations on A to be part of the non-matching
    suffix, but later operations on other qubits will continue to be put into
    the matching part (as long as those qubits have had no non-matching operation
    up to that point).
    """
    blocked_qubits: Set[cirq.Qid] = set()
    matching_prefix = circuits.Circuit()
    general_suffix = circuits.Circuit()
    for moment in circuit:
        matching_part = []
        general_part = []
        for op in moment:
            qs = set(op.qubits)
            if not predicate(op) or not qs.isdisjoint(blocked_qubits):
                blocked_qubits |= qs
            if qs.isdisjoint(blocked_qubits):
                matching_part.append(op)
            else:
                general_part.append(op)
        if matching_part:
            matching_prefix.append(circuits.Moment(matching_part))
        if general_part:
            general_suffix.append(circuits.Moment(general_part))
    return (matching_prefix, general_suffix)