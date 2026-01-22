from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_8_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(1, 0)
    qc.s(0)
    qc.sdg(1)
    qc.cx(1, 0)
    qc.sdg(0)
    qc.h(1)
    return qc