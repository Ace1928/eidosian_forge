from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_4_2():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    qc.cz(0, 1)
    return qc