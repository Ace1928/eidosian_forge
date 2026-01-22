from qiskit.circuit import QuantumCircuit, CircuitInstruction
def dagdependency_to_circuit(dagdependency):
    """Build a ``QuantumCircuit`` object from a ``DAGDependency``.

    Args:
        dagdependency (DAGDependency): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag dependency.
    """
    name = dagdependency.name or None
    circuit = QuantumCircuit(dagdependency.qubits, dagdependency.clbits, *dagdependency.qregs.values(), *dagdependency.cregs.values(), name=name)
    circuit.metadata = dagdependency.metadata
    circuit.calibrations = dagdependency.calibrations
    for node in dagdependency.topological_nodes():
        circuit._append(CircuitInstruction(node.op.copy(), node.qargs, node.cargs))
    return circuit