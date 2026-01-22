from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_4_4():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.s(0)
    qc.cz(0, 1)
    qc.sdg(0)
    qc.cz(0, 1)
    return qc