import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
def display_mapping(circuit: 'cirq.Circuit', initial_mapping: LogicalMapping) -> None:
    """Inserts display gates between moments to indicate the mapping throughout
    the circuit."""
    qubits = sorted(circuit.all_qubits())
    mapping = initial_mapping.copy()
    old_moments = circuit._moments
    gate = MappingDisplayGate((mapping.get(q) for q in qubits))
    new_moments = [circuits.Moment([gate(*qubits)])]
    for moment in old_moments:
        new_moments.append(moment)
        update_mapping(mapping, moment)
        gate = MappingDisplayGate((mapping.get(q) for q in qubits))
        new_moments.append(circuits.Moment([gate(*qubits)]))
    circuit._moments = new_moments