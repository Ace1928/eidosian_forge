from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_2_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.cz(0, 1)
    return qc